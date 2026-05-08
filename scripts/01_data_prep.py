#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_data_prep.py — 数据预处理
功能：加载原始数据 → jieba分词 → 去停用词 → 构建词汇表 → 编码保存
"""

import os
import re
import pickle
import jieba
from collections import Counter

# ---------- 路径配置 ----------
DATA_DIR = os.path.join('..', 'data', 'THUCNews', 'data')
PROCESSED_DIR = os.path.join('..', 'data', 'processed')
STOPWORDS_PATH = os.path.join('..', 'data', 'stopwords.txt')
OUTPUT_DIR = os.path.join('..', 'output')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 参数 ----------
MAX_VOCAB_SIZE = 50000
MIN_WORD_FREQ = 5
MAX_SEQ_LENGTH = 200

# 类别名（与 data_label 顺序一致，0-indexed）
CLASS_NAMES = [
    'finance', 'realty', 'stocks', 'education', 'science',
    'society', 'politics', 'sports', 'game', 'entertainment'
]


def load_stopwords(path):
    """加载停用词表，不存在则用内置基础停用词"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    print(f"  停用词表不存在 ({path})，使用内置基础停用词")
    return {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        '什么', '怎么', '这个', '那个', '因为', '所以', '如果', '虽然', '但是',
        '可以', '之', '与', '及', '等', '或', '被', '把', '对', '从', '以',
        '而', '但', '其', '中', '将', '已', '还', '又', '再', '才', '更',
        '能', '所', '为', '于', '向', '让', '比', '吗', '吧', '呢', '啊',
        '呀', '哦', '嗯', '哈', '嘛', '量', '此外', '本身', '比如', '并且',
        '不管', '不仅', '不论', '不只', '超过', '充满', '从而', '除了',
        '大约', '大量', '当时', '到处', '当中', '而且', '而是', '反而',
        '方面', '根据', '关于', '很多', '基于', '及其', '以来', '以及',
        '例如', '按照', '经过', '通过', '为了', '因此', '由于'
    }


def load_data(filepath):
    """加载 THUCNews 数据，返回 (texts, labels)"""
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return texts, labels


def clean_text(text):
    """保留中文字符、英文字母、数字"""
    text = re.sub(r'\s+', '', text)
    return re.sub(r'[^一-龥a-zA-Z0-9]', '', text)


def tokenize(texts, stopwords):
    """jieba 分词 + 去停用词"""
    result = []
    for text in texts:
        cleaned = clean_text(text)
        words = jieba.lcut(cleaned)
        words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
        result.append(words)
    return result


def build_vocab(tokenized_texts, max_size=MAX_VOCAB_SIZE, min_freq=MIN_WORD_FREQ):
    """构建词汇表，含 <PAD> 和 <UNK>"""
    counter = Counter()
    for words in tokenized_texts:
        counter.update(words)

    vocab = [w for w, c in counter.most_common(max_size) if c >= min_freq]
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for w in vocab:
        word2idx[w] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def encode_texts(tokenized_texts, word2idx, max_len=MAX_SEQ_LENGTH):
    """将词序列转成固定长度的索引序列"""
    unk_idx = word2idx['<UNK>']
    pad_idx = word2idx['<PAD>']
    encoded = []
    for words in tokenized_texts:
        indices = [word2idx.get(w, unk_idx) for w in words[:max_len]]
        if len(indices) < max_len:
            indices += [pad_idx] * (max_len - len(indices))
        encoded.append(indices)
    return encoded


def main():
    print('=' * 60)
    print('01 — 数据预处理')
    print('=' * 60)

    # 1. 停用词
    print('\n[1/5] 加载停用词…')
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f'  停用词数量: {len(stopwords)}')

    # 2. 加载原始数据
    print('\n[2/5] 加载原始数据…')
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, 'train.txt'))
    dev_texts, dev_labels = load_data(os.path.join(DATA_DIR, 'dev.txt'))
    test_texts, test_labels = load_data(os.path.join(DATA_DIR, 'test.txt'))
    print(f'  训练集: {len(train_texts)}')
    print(f'  验证集: {len(dev_texts)}')
    print(f'  测试集: {len(test_texts)}')

    # 3. 分词
    print('\n[3/5] jieba 分词 (耐心等待，约 2-3 分钟)…')
    all_texts = train_texts + dev_texts + test_texts
    all_tokenized = tokenize(all_texts, stopwords)
    n_train = len(train_texts)
    n_dev = len(dev_texts)
    train_tok = all_tokenized[:n_train]
    dev_tok = all_tokenized[n_train:n_train + n_dev]
    test_tok = all_tokenized[n_train + n_dev:]

    print(f'\n  分词样例:')
    print(f'  原文:  {train_texts[0][:60]}')
    print(f'  分词:  {train_tok[0][:15]}')

    # 4. 词汇表（仅基于训练集）
    print('\n[4/5] 构建词汇表…')
    word2idx, idx2word = build_vocab(train_tok)
    vocab_size = len(word2idx)
    print(f'  词汇表大小: {vocab_size}')

    # 5. 编码并保存
    print('\n[5/5] 编码序列 → 保存预处理数据…')
    train_enc = encode_texts(train_tok, word2idx)
    dev_enc = encode_texts(dev_tok, word2idx)
    test_enc = encode_texts(test_tok, word2idx)

    processed = {
        'train_texts': train_texts,
        'train_tokenized': train_tok,
        'train_encoded': train_enc,
        'train_labels': train_labels,
        'dev_texts': dev_texts,
        'dev_tokenized': dev_tok,
        'dev_encoded': dev_enc,
        'dev_labels': dev_labels,
        'test_texts': test_texts,
        'test_tokenized': test_tok,
        'test_encoded': test_enc,
        'test_labels': test_labels,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'class_names': CLASS_NAMES,
        'vocab_size': vocab_size,
        'max_seq_length': MAX_SEQ_LENGTH,
    }
    save_path = os.path.join(PROCESSED_DIR, 'processed.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(processed, f)
    print(f'  已保存至: {save_path}')

    # 统计摘要
    print('\n' + '=' * 60)
    print('预处理完成 — 统计摘要')
    print('=' * 60)
    print(f'  总样本数: {len(all_texts)}')
    print(f'  词汇表大小: {vocab_size}')
    print(f'  最大序列长度: {MAX_SEQ_LENGTH}')
    print(f'  类别数: {len(CLASS_NAMES)}')
    label_counts = Counter(train_labels)
    for lb in sorted(label_counts):
        print(f'    {CLASS_NAMES[lb]}: {label_counts[lb]}')


if __name__ == '__main__':
    main()
