#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_sentiment_analysis.py — 基于词典的情感分析
功能：使用 cnsenti（DUTIR 词典）对 THUCNews 数据进行情感分析
      统计各类别的正负情感分布
"""

import os
import pickle
import time
import random
import numpy as np
import pandas as pd
from collections import Counter
from cnsenti import Sentiment
import matplotlib
import matplotlib.font_manager as fm
for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 路径 ----------
PROCESSED_PATH = os.path.join('..', 'data', 'processed', 'processed.pkl')
FIGURE_DIR = os.path.join('..', 'output', 'figures')
TABLE_DIR = os.path.join('..', 'output', 'tables')
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# 配色
COLORS = ['#4DBBD5', '#F39B7F', '#00A087', '#91D1C2',
          '#8491B4', '#FCC5A1', '#E64B35', '#3C5488',
          '#7E6148', '#DC0000']


def analyze_sentiment_batch(texts, batch_size=500):
    """批量情感分析"""
    sent = Sentiment()
    results = []
    n = len(texts)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            # cnsenti 返回 {'words': int, 'sentences': int, 'pos': int, 'neg': int}
            res = sent.sentiment_count(text)
            results.append(res)
        if (i // batch_size) % 10 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f'  已处理 {i}/{n} ({i/n*100:.0f}%), 耗时 {elapsed:.0f}s')
    elapsed = time.time() - t0
    print(f'  完成 {n} 条, 总耗时 {elapsed:.0f}s ({elapsed/n*1000:.1f}ms/条)')
    return results


def classify_sentiment(pos_count, neg_count):
    """根据正负情感词数量判定情感极性"""
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


def plot_sentiment_distribution(sentiment_df, class_names, save_path):
    """情感极性分布堆叠柱状图"""
    class_order = sorted(sentiment_df['label'].unique())
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(class_order))
    width = 0.6
    labels_order = ['negative', 'neutral', 'positive']
    colors_map = {'negative': '#F39B7F', 'neutral': '#91D1C2', 'positive': '#4DBBD5'}

    bottom = np.zeros(len(class_order))
    for lbl in labels_order:
        counts = []
        for cl in class_order:
            subset = sentiment_df[sentiment_df['label'] == cl]
            counts.append((subset['sentiment'] == lbl).sum())
        ax.bar(x, counts, width, bottom=bottom, label=lbl, color=colors_map[lbl])
        bottom += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_xlabel('类别')
    ax.set_ylabel('样本数')
    ax.set_title('各类别情感极性分布')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')


def plot_sentiment_scores_distribution(sentiment_df, class_names, save_path):
    """情感得分箱线图（各类别正负情感词数量）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, score_col, title in zip(axes, ['pos', 'neg'],
                                      ['Positive Word Count', 'Negative Word Count']):
        data = []
        labels = []
        for i, cn in enumerate(class_names):
            vals = sentiment_df[sentiment_df['label'] == i][score_col].values
            data.append(vals)
            labels.append(cn)
        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title(title)
        ax.set_ylabel('词数')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')


def main():
    print('=' * 60)
    print('05 — 基于词典的情感分析')
    print('=' * 60)

    # 1. 加载数据
    print('\n[1/4] 加载数据…')
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']

    # 仅对测试集做情感分析（10000 条，平衡各约 1000/类）
    test_texts = data['test_texts']
    test_labels = data['test_labels']
    print(f'  测试集: {len(test_texts)} 条')

    # 2. 情感分析
    print('\n[2/4] 运行 cnsenti 情感分析…')
    sentiment_results = analyze_sentiment_batch(test_texts, batch_size=500)

    # 3. 整理结果
    print('\n[3/4] 整理结果…')
    rows = []
    for i, res in enumerate(sentiment_results):
        rows.append({
            'text': test_texts[i][:50],
            'label': test_labels[i],
            'class_name': class_names[test_labels[i]],
            'pos': res['pos'],
            'neg': res['neg'],
            'sentiment': classify_sentiment(res['pos'], res['neg']),
        })
    sentiment_df = pd.DataFrame(rows)

    # 各类别情感统计
    print('\n  各类别情感统计:')
    print(f'  {"类别":<15} {"Positive":>8} {"Neutral":>8} {"Negative":>8} {"Pos%"} ')
    for cn in class_names:
        sub = sentiment_df[sentiment_df['class_name'] == cn]
        pos = (sub['sentiment'] == 'positive').sum()
        neg = (sub['sentiment'] == 'negative').sum()
        neu = (sub['sentiment'] == 'neutral').sum()
        print(f'  {cn:<15} {pos:>8} {neu:>8} {neg:>8} {pos/(pos+neg+neu)*100:>5.1f}%')

    # 4. 保存图表
    print('\n[4/4] 生成图表…')
    plot_sentiment_distribution(sentiment_df, class_names,
                                os.path.join(FIGURE_DIR, 'fig_sentiment_distribution.png'))
    plot_sentiment_scores_distribution(sentiment_df, class_names,
                                       os.path.join(FIGURE_DIR, 'fig_sentiment_scores.png'))

    # 保存数据
    sentiment_df.to_pickle(os.path.join('..', 'data', 'processed', 'sentiment_results.pkl'))
    print(f'  情感结果保存至: data/processed/sentiment_results.pkl')

    # 摘要统计
    total_pos = (sentiment_df['sentiment'] == 'positive').sum()
    total_neg = (sentiment_df['sentiment'] == 'negative').sum()
    total_neu = (sentiment_df['sentiment'] == 'neutral').sum()
    print('\n' + '=' * 60)
    print('情感分析完成 — 测试集摘要')
    print('=' * 60)
    print(f'  总样本: {len(sentiment_df)}')
    print(f'  正面: {total_pos} ({total_pos/len(sentiment_df)*100:.1f}%)')
    print(f'  中性: {total_neu} ({total_neu/len(sentiment_df)*100:.1f}%)')
    print(f'  负面: {total_neg} ({total_neg/len(sentiment_df)*100:.1f}%)')


if __name__ == '__main__':
    main()
