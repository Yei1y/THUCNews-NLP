#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_lstm_classifier.py — LSTM 文本分类 + Naive Bayes 基线
功能：LSTM 模型搭建、训练、调参、评估；NB+TF-IDF 基线对比
"""

import os
import sys
import pickle
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
import matplotlib.font_manager as fm
for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 路径 ----------
PROCESSED_PATH = os.path.join('..', 'data', 'processed', 'processed.pkl')
EMB_PATH = os.path.join('..', 'output', 'models', 'embedding_matrix.npy')
FIGURE_DIR = os.path.join('..', 'output', 'figures')
TABLE_DIR = os.path.join('..', 'output', 'tables')
MODEL_DIR = os.path.join('..', 'output', 'models')
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------- 超参数 ----------
HIDDEN_SIZES = [64, 128]
DROPOUTS = [0.5]
BATCH_SIZE = 128
LR = 0.01
NUM_EPOCHS = 20
PATIENCE = 4
SUBSAMPLE = 3000   # 每类训练样本数
SEQ_LENGTH = 50    # 截断序列长度（原 200 太长，均值仅 8）

# ---------- 设备 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, num_classes, dropout, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                     # (B, L, emb_dim)
        lstm_out, _ = self.lstm(emb)                # (B, L, hidden_size)
        # 对非填充位置做平均池化
        mask = (x != 0).unsqueeze(-1).float()        # (B, L, 1)
        masked_out = lstm_out * mask                 # (B, L, hidden_size)
        sum_out = masked_out.sum(dim=1)              # (B, hidden_size)
        len_out = mask.sum(dim=1).clamp(min=1)       # (B, 1)
        pooled = sum_out / len_out                   # (B, hidden_size)
        out = self.dropout(pooled)
        return self.fc(out)                         # (B, num_classes)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, total_correct / total, all_preds, all_labels


def plot_training_curves(history, config, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ['loss', 'acc'],
                                   [f'Loss (hs={config[0]}, dp={config[1]})',
                                    f'Accuracy (hs={config[0]}, dp={config[1]})']):
        ax.plot(history[f'train_{metric}'], label='Train', color='#4DBBD5')
        ax.plot(history[f'val_{metric}'], label='Val', color='#F39B7F')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend()
    plt.suptitle('LSTM Training History', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_lstm_experiment(train_enc, train_labels, dev_enc, dev_labels,
                         test_enc, test_labels, emb_matrix, class_names):
    """在不同超参数下训练 LSTM，返回最佳模型结果"""
    # 转换为 DataLoader（截断序列长度）
    def make_loader(x, y, shuffle=False):
        x = [seq[:SEQ_LENGTH] for seq in x]
        t = TensorDataset(torch.LongTensor(x), torch.LongTensor(y))
        return DataLoader(t, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_enc, train_labels, shuffle=True)
    dev_loader = make_loader(dev_enc, dev_labels)
    test_loader = make_loader(test_enc, test_labels)

    vocab_size, emb_dim = emb_matrix.shape
    pretrained_emb = torch.FloatTensor(emb_matrix)
    num_classes = len(class_names)

    best_val_acc = 0
    best_model = None
    best_config = None
    best_history = None
    results = []

    for hidden_size in HIDDEN_SIZES:
        for dropout in DROPOUTS:
            config_name = f'LSTM_h{hidden_size}_d{dropout}'
            print(f'\n--- Config: {config_name} ---')

            model = LSTMClassifier(vocab_size, emb_dim, hidden_size, 1,
                                    num_classes, dropout, pretrained_emb).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch = 0

            t0 = time.time()
            for epoch in range(1, NUM_EPOCHS + 1):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, _, _ = evaluate(model, dev_loader, criterion)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print(f'  E{epoch:2d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
                      f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f'  Early stopping @ epoch {epoch}, best epoch={best_epoch}')
                        break

            elapsed = time.time() - t0
            print(f'  Time: {elapsed:.1f}s, Best Val Acc: {max(history["val_acc"]):.4f}')

            # 恢复最佳模型
            model.load_state_dict(best_model_state)

            # Test evaluation
            _, test_acc, test_preds, test_true = evaluate(model, test_loader, criterion)
            test_f1_macro = f1_score(test_true, test_preds, average='macro')
            test_f1_weighted = f1_score(test_true, test_preds, average='weighted')
            print(f'  Test | acc={test_acc:.4f} f1_macro={test_f1_macro:.4f}')

            # 保存训练曲线
            plot_training_curves(history, (hidden_size, dropout),
                                 os.path.join(FIGURE_DIR, f'lstm_{config_name}_curves.png'))

            # 记录结果
            res = {
                'config': config_name,
                'hidden_size': hidden_size,
                'dropout': dropout,
                'test_acc': round(test_acc, 4),
                'test_f1_macro': round(test_f1_macro, 4),
                'test_f1_weighted': round(test_f1_weighted, 4),
                'best_val_acc': round(max(history['val_acc']), 4),
                'best_epoch': best_epoch,
                'time': round(elapsed, 1),
            }
            results.append(res)

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                best_model = model
                best_config = config_name
                best_model_state_dict = best_model_state

    # 保存最佳模型
    torch.save(best_model_state_dict, os.path.join(MODEL_DIR, 'lstm_best.pt'))
    print(f'\n最佳模型: {best_config}, test_acc={best_val_acc:.4f}')

    return results, best_config, best_model_state_dict


def run_naive_bayes(train_texts, train_labels, test_texts, test_labels, class_names):
    """Naive Bayes + TF-IDF 基线"""
    print('\n' + '=' * 60)
    print('Naive Bayes + TF-IDF 基线')
    print('=' * 60)

    t0 = time.time()
    vectorizer = CountVectorizer(max_features=10000, analyzer='word', token_pattern=r'(?u)\b\w+\b')
    X_train_counts = vectorizer.fit_transform(train_texts)
    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train_counts)
    X_test = vectorizer.transform(test_texts)
    X_test = tfidf.transform(X_test)
    print(f'  TF-IDF 特征维度: {X_train.shape[1]}')

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, train_labels)
    preds = nb.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro')
    f1_weighted = f1_score(test_labels, preds, average='weighted')

    print(f'  Time: {elapsed:.1f}s')
    print(f'  Test acc={acc:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}')

    # 混淆矩阵
    plot_confusion_matrix(test_labels, preds, class_names,
                          os.path.join(FIGURE_DIR, 'nb_confusion_matrix.png'))

    # 分类报告
    report = classification_report(test_labels, preds, target_names=class_names, output_dict=True)
    return {
        'model': nb,
        'vectorizer': vectorizer,
        'tfidf': tfidf,
        'test_acc': round(acc, 4),
        'test_f1_macro': round(f1_macro, 4),
        'test_f1_weighted': round(f1_weighted, 4),
        'time': round(elapsed, 1),
        'preds': preds,
        'report': report,
    }


def save_results_table(lstm_results, nb_result, class_names, path):
    """保存结果对比表格"""
    lines = ['# 分类结果对比\n']
    lines.append('| Model | Test Acc | F1 (macro) | F1 (weighted) | Time (s) |')
    lines.append('|-------|----------|------------|---------------|----------|')

    for r in lstm_results:
        lines.append(f"| {r['config']} | {r['test_acc']} | {r['test_f1_macro']} | "
                     f"{r['test_f1_weighted']} | {r['time']} |")

    lines.append(f"| Naive Bayes+TF-IDF | {nb_result['test_acc']} | "
                 f"{nb_result['test_f1_macro']} | {nb_result['test_f1_weighted']} | "
                 f"{nb_result['time']} |")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  结果表保存至: {path}')


def main():
    print('=' * 60)
    print('04 — LSTM 文本分类 + Naive Bayes 基线')
    print('=' * 60)

    # 1. 加载数据
    print('\n[1/4] 加载预处理数据…')
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    train_enc, train_labels = data['train_encoded'], data['train_labels']
    dev_enc, dev_labels = data['dev_encoded'], data['dev_labels']
    test_enc, test_labels = data['test_encoded'], data['test_labels']
    train_texts, test_texts = data['train_texts'], data['test_texts']
    class_names = data['class_names']
    print(f'  训练: {len(train_enc)} | 验证: {len(dev_enc)} | 测试: {len(test_enc)}')

    # 2. 加载词向量
    print('\n[2/4] 加载预训练词向量…')
    emb_matrix = np.load(EMB_PATH)
    print(f'  嵌入矩阵: {emb_matrix.shape}')

    # 3. 对训练集按类别等比例采样（加速 LSTM 训练）
    print('\n[3/4] LSTM 训练与调参…')
    train_sub_enc, train_sub_labels = [], []
    for lb in range(len(class_names)):
        indices = [i for i, l in enumerate(train_labels) if l == lb]
        chosen = random.sample(indices, min(SUBSAMPLE, len(indices)))
        train_sub_enc.extend([train_enc[i] for i in chosen])
        train_sub_labels.extend([train_labels[i] for i in chosen])
    print(f'  LSTM 训练子集: {len(train_sub_enc)} 条 ({SUBSAMPLE}/类)')

    lstm_results, best_config, best_state = run_lstm_experiment(
        train_sub_enc, train_sub_labels, dev_enc, dev_labels,
        test_enc, test_labels, emb_matrix, class_names
    )

    # 最佳模型在测试集上的详细结果
    criterion = nn.CrossEntropyLoss()
    vocab_size, emb_dim = emb_matrix.shape
    best_hs = [r for r in lstm_results if r['config'] == best_config][0]
    best_model = LSTMClassifier(vocab_size, emb_dim, best_hs['hidden_size'], 1,
                                 len(class_names), best_hs['dropout'],
                                 torch.FloatTensor(emb_matrix)).to(device)
    best_model.load_state_dict(best_state)

    def make_loader(x, y):
        t = TensorDataset(torch.LongTensor(x), torch.LongTensor(y))
        return DataLoader(t, batch_size=BATCH_SIZE)

    _, _, test_preds, test_true = evaluate(best_model, make_loader(test_enc, test_labels), criterion)
    plot_confusion_matrix(test_true, test_preds, class_names,
                          os.path.join(FIGURE_DIR, f'lstm_{best_config}_confusion_matrix.png'))

    # 4. Naive Bayes 基线
    print('\n[4/4] Naive Bayes + TF-IDF 基线…')
    nb_result = run_naive_bayes(train_texts, train_labels, test_texts, test_labels, class_names)

    # 5. 保存结果
    print('\n--- 保存结果 ---')
    save_results_table(lstm_results, nb_result, class_names,
                       os.path.join(TABLE_DIR, 'classification_results.md'))

    print('\n' + '=' * 60)
    print('分类实验完成')
    print('=' * 60)
    print(f'  LSTM 最佳: {best_config}')
    print(f'    Test Acc: {best_hs["test_acc"]}')
    print(f'    Test F1 (macro): {best_hs["test_f1_macro"]}')
    print(f'  Naive Bayes:')
    print(f'    Test Acc: {nb_result["test_acc"]}')
    print(f'    Test F1 (macro): {nb_result["test_f1_macro"]}')


if __name__ == '__main__':
    main()
