#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_baseline_diagnosis.py — 基线公平性与结论可核验性诊断

功能（只计算与落盘，不在代码里写分析结论）：
  1. 诊断 Naive Bayes 基线的分词器配置：原始文本 vs jieba 分词后的 token 数
  2. 在同一训练子集（与 LSTM 相同的 3,000/类 采样）与全量训练集上重算 NB
  3. 从 output/models/lstm_best.pt 复算 LSTM 各类别准确率
  4. 分解"中性"情感样本的构成（无情感词 vs 正负打平）
  5. 计算类别准确率与正面情感占比的相关系数

产物：
  output/tables/baseline_diagnosis.md
  output/figures/fig_nb_tokenization_diagnosis.png
  output/figures/fig_lstm_per_class_accuracy.png
"""

import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
import matplotlib.font_manager as fm
for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(ROOT, 'data', 'processed', 'processed.pkl')
SENTIMENT_PATH = os.path.join(ROOT, 'data', 'processed', 'sentiment_results.pkl')
EMB_PATH = os.path.join(ROOT, 'output', 'models', 'embedding_matrix.npy')
LSTM_PATH = os.path.join(ROOT, 'output', 'models', 'lstm_best.pt')
FIGURE_DIR = os.path.join(ROOT, 'output', 'figures')
TABLE_DIR = os.path.join(ROOT, 'output', 'tables')
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ---------- 与 04 保持一致的设置 ----------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

SUBSAMPLE = 3000
SEQ_LENGTH = 50
MAX_FEATURES = 10000

# 论文原始设定：未分词文本 + 把连续中英文数字串当作一个 token
RAW_TOKEN_PATTERN = r'(?u)\b\w+\b'

COLORS = ['#4DBBD5', '#F39B7F', '#00A087', '#91D1C2',
          '#8491B4', '#FCC5A1', '#E64B35', '#3C5488',
          '#7E6148', '#DC0000']


class LSTMClassifier(nn.Module):
    """与 04_lstm_classifier.py 完全一致的模型定义，用于复算"""

    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers,
                 num_classes, dropout, pretrained_emb):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_emb, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(self.dropout(pooled))


def nb_pipeline(train_docs, train_labels, test_docs, test_labels, token_pattern=None):
    """构造 TF-IDF 特征并拟合 MultinomialNB，返回评估指标与预测"""
    kwargs = {'max_features': MAX_FEATURES}
    if token_pattern is not None:
        kwargs.update({'analyzer': 'word', 'token_pattern': token_pattern})
    vectorizer = CountVectorizer(**kwargs)
    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(vectorizer.fit_transform(train_docs))
    X_test = tfidf.transform(vectorizer.transform(test_docs))

    t0 = time.time()
    nb = MultinomialNB(alpha=1.0).fit(X_train, train_labels)
    preds = nb.predict(X_test)
    elapsed = time.time() - t0

    return {
        'acc': round(float(accuracy_score(test_labels, preds)), 4),
        'f1_macro': round(float(f1_score(test_labels, preds, average='macro')), 4),
        'f1_weighted': round(float(f1_score(test_labels, preds, average='weighted')), 4),
        'n_features': int(X_train.shape[1]),
        'fit_secs': round(elapsed, 1),
        'preds': preds,
    }


def plot_tokenization_diagnosis(raw_tokens, jieba_tokens, broken_preds,
                                fixed_preds, class_names, save_path):
    """左：每条文本的 token 数；右：两种设定的预测类别分布"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    positions = [1, 2]
    bp = ax.boxplot([raw_tokens, jieba_tokens], positions=positions,
                    widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], [COLORS[1], COLORS[0]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'原始文本\n$\\bar x$={raw_tokens.mean():.2f}',
                        f'jieba 分词\n$\\bar x$={jieba_tokens.mean():.2f}'])
    ax.set_ylabel('每条文本的 token 数')
    ax.set_title('CountVectorizer 实际切出的 token 数')

    ax = axes[1]
    x = np.arange(len(class_names))
    width = 0.38
    broken_counts = [int((broken_preds == i).sum()) for i in range(len(class_names))]
    fixed_counts = [int((fixed_preds == i).sum()) for i in range(len(class_names))]
    ax.bar(x - width / 2, broken_counts, width, label='原始文本设定', color=COLORS[1])
    ax.bar(x + width / 2, fixed_counts, width, label='jieba 分词设定', color=COLORS[0])
    ax.axhline(1000, color='grey', ls='--', lw=1, label='测试集每类真实样本数 = 1000')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylabel('预测为该类的样本数')
    ax.set_title('NB 预测分布（测试集 n = 10,000）')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')


def plot_per_class_accuracy(per_class_acc, pos_pct, class_names, corr, save_path):
    """左：LSTM 各类别准确率；右：准确率 vs 正面情感占比散点"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    order = sorted(class_names, key=lambda c: per_class_acc[c])
    ax = axes[0]
    vals = [per_class_acc[c] for c in order]
    bars = ax.barh(range(len(order)), vals, color=COLORS[0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel('LSTM 各类别准确率')
    ax.set_title('LSTM 分类准确率的类别差异')
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8)

    ax = axes[1]
    xs = [per_class_acc[c] for c in class_names]
    ys = [pos_pct[c] for c in class_names]
    ax.scatter(xs, ys, s=70, color=COLORS[2], zorder=3)
    for c, xx, yy in zip(class_names, xs, ys):
        ax.annotate(c, (xx, yy), textcoords='offset points', xytext=(6, 4), fontsize=8)
    if len(set(xs)) > 1:
        k, b = np.polyfit(xs, ys, 1)
        xr = np.linspace(min(xs), max(xs), 50)
        ax.plot(xr, k * xr + b, color=COLORS[1], ls='--', lw=1.5,
                label=f'OLS: r = {corr["pearson_r"]:.3f}, p = {corr["pearson_p"]:.3f}')
    ax.set_xlabel('LSTM 类别准确率')
    ax.set_ylabel('正面情感样本占比 (%)')
    ax.set_title('类别准确率与正面情感占比的关系')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')


def main():
    print('=' * 60)
    print('07 — 基线公平性与结论可核验性诊断')
    print('=' * 60)

    # ---------- 1. 加载数据 ----------
    print('\n[1/6] 加载预处理数据…')
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']
    train_tok, test_tok = data['train_tokenized'], data['test_tokenized']
    n_classes = len(class_names)

    # ---------- 2. 复现 04 的训练子集（3,000/类） ----------
    print('\n[2/6] 复现 LSTM 训练子集 (3,000/类)…')
    sub_idx = []
    for lb in range(n_classes):
        indices = [i for i, l in enumerate(train_labels) if l == lb]
        sub_idx.extend(random.sample(indices, min(SUBSAMPLE, len(indices))))
    print(f'  子集规模: {len(sub_idx)}')

    # ---------- 3. 分词器诊断 ----------
    print('\n[3/6] 诊断 NB 基线的分词器…')
    probe = CountVectorizer(max_features=MAX_FEATURES, analyzer='word',
                            token_pattern=RAW_TOKEN_PATTERN)
    probe.fit([train_texts[i] for i in sub_idx])
    analyze = probe.build_analyzer()
    raw_tokens = np.array([len(analyze(t)) for t in train_texts[:5000]])
    jieba_tokens = np.array([len(t) for t in train_tok[:5000]])
    print(f'  原始文本 token 数: 均值={raw_tokens.mean():.2f}, 中位数={np.median(raw_tokens):.0f}')
    print(f'  jieba 分词 token 数: 均值={jieba_tokens.mean():.2f}, 中位数={np.median(jieba_tokens):.0f}')

    # ---------- 4. 三种 NB 设定 ----------
    print('\n[4/6] 重算 Naive Bayes 基线…')
    sub_texts = [train_texts[i] for i in sub_idx]
    sub_labels = [train_labels[i] for i in sub_idx]
    sub_jieba = [' '.join(train_tok[i]) for i in sub_idx]
    test_jieba = [' '.join(t) for t in test_tok]

    nb_raw_sub = nb_pipeline(sub_texts, sub_labels, test_texts, test_labels, RAW_TOKEN_PATTERN)
    nb_jieba_sub = nb_pipeline(sub_jieba, sub_labels, test_jieba, test_labels)
    nb_jieba_full = nb_pipeline([' '.join(t) for t in train_tok], train_labels,
                                test_jieba, test_labels)
    print(f'  原始文本 / 子集训练 : acc={nb_raw_sub["acc"]}')
    print(f'  jieba 分词 / 子集训练: acc={nb_jieba_sub["acc"]}')
    print(f'  jieba 分词 / 全量训练: acc={nb_jieba_full["acc"]}')

    # ---------- 5. 复算 LSTM 各类别准确率 ----------
    print('\n[5/6] 复算 LSTM 各类别准确率…')
    emb_matrix = np.load(EMB_PATH)
    model = LSTMClassifier(emb_matrix.shape[0], emb_matrix.shape[1], 64, 1,
                           n_classes, 0.5, torch.FloatTensor(emb_matrix))
    model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
    model.eval()

    x = torch.LongTensor([s[:SEQ_LENGTH] for s in data['test_encoded']])
    loader = DataLoader(TensorDataset(x, torch.LongTensor(test_labels)), batch_size=256)
    lstm_preds = []
    with torch.no_grad():
        for xb, _ in loader:
            lstm_preds.extend(model(xb).argmax(1).numpy())
    lstm_preds = np.array(lstm_preds)
    y_true = np.array(test_labels)

    lstm_acc = round(float((lstm_preds == y_true).mean()), 4)
    lstm_f1 = round(float(f1_score(y_true, lstm_preds, average='macro')), 4)
    print(f'  LSTM 复算: acc={lstm_acc}, f1_macro={lstm_f1}')

    cm = confusion_matrix(y_true, lstm_preds)
    per_class_acc = {class_names[i]: round(float(cm[i, i] / cm[i].sum()), 4)
                     for i in range(n_classes)}
    top_confusions = sorted(
        [[class_names[i], class_names[j], int(cm[i, j])]
         for i in range(n_classes) for j in range(n_classes) if i != j and cm[i, j] > 0],
        key=lambda r: -r[2])[:8]

    # ---------- 6. 情感分解与关联 ----------
    print('\n[6/6] 分解中性样本并计算关联…')
    sdf = pd.read_pickle(SENTIMENT_PATH)
    n_senti = len(sdf)
    zero = int(((sdf['pos'] == 0) & (sdf['neg'] == 0)).sum())
    neu_total = int((sdf['sentiment'] == 'neutral').sum())
    neu_tie = neu_total - zero

    pos_pct = {class_names[i]: round(float(
        (sdf[sdf['label'] == i]['sentiment'] == 'positive').mean() * 100), 1)
        for i in range(n_classes)}

    a = np.array([per_class_acc[c] for c in class_names])
    p = np.array([pos_pct[c] for c in class_names])
    corr = {'pearson_r': round(float(pearsonr(a, p)[0]), 4),
            'pearson_p': round(float(pearsonr(a, p)[1]), 4),
            'spearman_rho': round(float(spearmanr(a, p)[0]), 4),
            'spearman_p': round(float(spearmanr(a, p)[1]), 4)}
    print(f'  无情感词样本: {zero}/{n_senti} ({zero / n_senti * 100:.1f}%)')
    print(f'  准确率 vs 正面占比: r={corr["pearson_r"]}, p={corr["pearson_p"]}')

    # ---------- 落盘 ----------
    plot_tokenization_diagnosis(raw_tokens, jieba_tokens, nb_raw_sub['preds'],
                                nb_jieba_sub['preds'], class_names,
                                os.path.join(FIGURE_DIR, 'fig_nb_tokenization_diagnosis.png'))
    plot_per_class_accuracy(per_class_acc, pos_pct, class_names, corr,
                            os.path.join(FIGURE_DIR, 'fig_lstm_per_class_accuracy.png'))

    lines = []
    lines.append('# 基线公平性与结论核验诊断\n')
    lines.append('## 1. NB 基线的分词器配置\n')
    lines.append('| 设定 | 每条文本 token 数（均值/中位数） | TF-IDF 特征数 | 测试准确率 | F1 (macro) | 拟合耗时 |')
    lines.append('|------|--------------------------------|---------------|------------|------------|----------|')
    lines.append(f'| 原始文本 + `\\b\\w+\\b`（论文设定） | {raw_tokens.mean():.2f} / {np.median(raw_tokens):.0f} | '
                 f'{nb_raw_sub["n_features"]} | {nb_raw_sub["acc"]:.4f} | {nb_raw_sub["f1_macro"]:.4f} | '
                 f'{nb_raw_sub["fit_secs"]}s |')
    lines.append(f'| jieba 分词，子集训练 n={len(sub_idx)} | {jieba_tokens.mean():.2f} / '
                 f'{np.median(jieba_tokens):.0f} | {nb_jieba_sub["n_features"]} | '
                 f'{nb_jieba_sub["acc"]:.4f} | {nb_jieba_sub["f1_macro"]:.4f} | {nb_jieba_sub["fit_secs"]}s |')
    lines.append(f'| jieba 分词，全量训练 n={len(train_texts)} | {jieba_tokens.mean():.2f} / '
                 f'{np.median(jieba_tokens):.0f} | {nb_jieba_full["n_features"]} | '
                 f'{nb_jieba_full["acc"]:.4f} | {nb_jieba_full["f1_macro"]:.4f} | {nb_jieba_full["fit_secs"]}s |')
    lines.append('')
    lines.append('## 2. NB 预测分布（测试集 n = 10,000，每类真实 1,000）\n')
    lines.append('| 类别 | 原始文本设定 | jieba 分词设定 |')
    lines.append('|------|--------------|----------------|')
    for i, c in enumerate(class_names):
        lines.append(f'| {c} | {int((nb_raw_sub["preds"] == i).sum())} | '
                     f'{int((nb_jieba_sub["preds"] == i).sum())} |')
    lines.append('')
    lines.append('## 3. LSTM 各类别准确率（由 `lstm_best.pt` 复算）\n')
    lines.append(f'整体：acc = {lstm_acc:.4f}，F1 (macro) = {lstm_f1:.4f}\n')
    lines.append('| 类别 | 准确率 | 正面情感占比 (%) |')
    lines.append('|------|--------|------------------|')
    for c in class_names:
        lines.append(f'| {c} | {per_class_acc[c]:.4f} | {pos_pct[c]:.1f} |')
    lines.append('')
    lines.append('错误最集中的类别对：\n')
    lines.append('| 真实类别 | 预测类别 | 样本数 |')
    lines.append('|----------|----------|--------|')
    for t, p_, n in top_confusions:
        lines.append(f'| {t} | {p_} | {n} |')
    lines.append('')
    lines.append('## 4. 类别准确率与正面情感占比的关联\n')
    lines.append(f'- Pearson r = {corr["pearson_r"]}（p = {corr["pearson_p"]}）')
    lines.append(f'- Spearman ρ = {corr["spearman_rho"]}（p = {corr["spearman_p"]}）')
    lines.append('')
    lines.append('## 5. "中性"情感样本的构成\n')
    lines.append(f'- 测试集总数：{n_senti}')
    lines.append(f'- 判定为中性：{neu_total}（{neu_total / n_senti * 100:.1f}%）')
    lines.append(f'- 其中无任何情感词：{zero}（占中性样本 {zero / neu_total * 100:.1f}%，'
                 f'占测试集 {zero / n_senti * 100:.1f}%）')
    lines.append(f'- 其中正负情感词数相等：{neu_tie}（占中性样本 {neu_tie / neu_total * 100:.1f}%）')
    lines.append('')

    table_path = os.path.join(TABLE_DIR, 'baseline_diagnosis.md')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  结果表保存至: {table_path}')

    print('\n' + '=' * 60)
    print('诊断完成')
    print('=' * 60)


if __name__ == '__main__':
    main()