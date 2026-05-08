#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_word2vec.py — 训练 Word2Vec 词向量
功能：在预处理后的分词文本上训练 Word2Vec 模型，保存供 LSTM 使用
"""

import os
import pickle
import time
import random
import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# ---------- 路径 ----------
PROCESSED_PATH = os.path.join('..', 'data', 'processed', 'processed.pkl')
MODEL_DIR = os.path.join('..', 'output', 'models')
FIGURE_DIR = os.path.join('..', 'output', 'figures')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ---------- Word2Vec 超参数 ----------
W2V_PARAMS = {
    'vector_size': 200,
    'window': 5,
    'min_count': 5,
    'workers': 4,
    'sg': 1,          # Skip-gram (1) > CBOW (0) for smaller datasets
    'hs': 0,          # Negative sampling
    'negative': 10,
    'sample': 1e-4,
    'epochs': 10,
}


def main():
    print('=' * 60)
    print('03 — 训练 Word2Vec 词向量')
    print('=' * 60)

    # 1. 加载分词数据
    print('\n[1/3] 加载预处理数据…')
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    train_tok = data['train_tokenized']
    print(f'  训练句子数: {len(train_tok)}')
    print(f'  Word2Vec 参数: {W2V_PARAMS}')

    # 2. 训练
    print('\n[2/3] 训练 Word2Vec…')
    t0 = time.time()
    model = Word2Vec(sentences=train_tok, **W2V_PARAMS)
    elapsed = time.time() - t0
    print(f'  训练耗时: {elapsed:.1f}s')
    print(f'  词向量数: {len(model.wv)}')
    print(f'  向量维度: {model.wv.vector_size}')

    # 3. 保存模型
    print('\n[3/3] 保存模型…')
    model_path = os.path.join(MODEL_DIR, 'word2vec.model')
    model.save(model_path)
    kv_path = os.path.join(MODEL_DIR, 'word2vec.kv')
    model.wv.save(kv_path)
    print(f'  模型保存至: {model_path}')
    print(f'  向量保存至: {kv_path}')

    # 构建嵌入矩阵（供 LSTM 使用）
    word2idx = data['word2idx']
    vocab_size = data['vocab_size']
    emb_dim = W2V_PARAMS['vector_size']

    emb_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, emb_dim)).astype(np.float32)
    found = 0
    for word, idx in word2idx.items():
        if word in model.wv:
            emb_matrix[idx] = model.wv[word]
            found += 1
    emb_path = os.path.join(MODEL_DIR, 'embedding_matrix.npy')
    np.save(emb_path, emb_matrix)
    print(f'  嵌入矩阵保存至: {emb_path}')
    print(f'  词覆盖: {found}/{vocab_size} ({found/vocab_size*100:.1f}%)')

    # 输出相似词示例
    print('\n  词向量相似性示例:')
    sample_words = ['中国', '公司', '市场', '教育', '游戏']
    for w in sample_words:
        if w in model.wv:
            similar = [s for s, _ in model.wv.most_similar(w, topn=5)]
            print(f'    "{w}" → {similar}')

    print('\n' + '=' * 60)
    print('Word2Vec 训练完成')
    print('=' * 60)


if __name__ == '__main__':
    main()
