#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_comparison_viz.py — 综合对比可视化与论文用图
功能：汇总分类与情感分析结果，生成论文所需的对比图表
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
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
SENTIMENT_PATH = os.path.join('..', 'data', 'processed', 'sentiment_results.pkl')
FIGURE_DIR = os.path.join('..', 'output', 'figures')
TABLE_DIR = os.path.join('..', 'output', 'tables')
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# 学术蓝橙配色（扩展至 10 色）
COLORS = ['#4DBBD5', '#F39B7F', '#00A087', '#91D1C2',
          '#8491B4', '#FCC5A1', '#E64B35', '#3C5488',
          '#7E6148', '#DC0000']


def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 10, 'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    })


def plot_model_comparison(results, nb_result, save_path):
    """LSTM 各配置 vs NB 准确率对比"""
    models = [r['config'] for r in results] + ['Naive Bayes']
    accs = [r['test_acc'] for r in results] + [nb_result['test_acc']]
    f1s = [r['test_f1_macro'] for r in results] + [nb_result['test_f1_macro']]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label='Accuracy', color='#4DBBD5')
    bars2 = ax.bar(x + width / 2, f1s, width, label='F1 (macro)', color='#F39B7F')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy vs F1 Macro')
    ax.legend()
    ax.set_ylim(0, 1)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')


def plot_sentiment_by_category(sentiment_df, class_names, save_path):
    """各类别正负情感平均词数对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names))
    width = 0.35

    pos_means = sentiment_df.groupby('label')['pos'].mean()
    neg_means = sentiment_df.groupby('label')['neg'].mean()

    bars1 = ax.bar(x - width / 2, pos_means.values, width, label='Positive Words', color='#4DBBD5')
    bars2 = ax.bar(x + width / 2, neg_means.values, width, label='Negative Words', color='#F39B7F')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylabel('Average Word Count')
    ax.set_title('Average Sentiment Word Count by Category')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: {save_path}')

    return pos_means, neg_means


def main():
    print('=' * 60)
    print('06 — 综合对比可视化与论文用图')
    print('=' * 60)

    set_style()

    # 1. 加载分类结果表
    print('\n[1/4] 加载分类结果…')
    results_path = os.path.join(TABLE_DIR, 'classification_results.md')
    if not os.path.exists(results_path):
        print('  ! 分类结果表不存在，请先运行 04_lstm_classifier.py')
        return

    # 手动解析 markdown 表
    with open(results_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    nb_result = {}
    for line in lines[2:]:
        if '|' not in line or line.strip().startswith('|--'):
            continue
        cols = [c.strip() for c in line.strip().split('|')[1:-1]]
        if len(cols) >= 5:
            if cols[0].startswith('LSTM'):
                results.append({
                    'config': cols[0],
                    'test_acc': float(cols[1]),
                    'test_f1_macro': float(cols[2]),
                    'test_f1_weighted': float(cols[3]),
                    'time': float(cols[4]),
                })
            elif 'Naive' in cols[0]:
                nb_result = {
                    'test_acc': float(cols[1]),
                    'test_f1_macro': float(cols[2]),
                    'test_f1_weighted': float(cols[3]),
                    'time': float(cols[4]),
                }
    print(f'  LSTM 配置数: {len(results)}')
    print(f'  NB baseline: acc={nb_result.get("test_acc", "N/A")}')

    # 2. 加载情感分析结果
    print('\n[2/4] 加载情感分析结果…')
    if os.path.exists(SENTIMENT_PATH):
        sentiment_df = pd.read_pickle(SENTIMENT_PATH)
        print(f'  样本数: {len(sentiment_df)}')
    else:
        print('  ! 情感分析结果不存在，请先运行 05_sentiment_analysis.py')
        return

    # 3. 加载类别名
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']

    # 4. 生成论文用图
    print('\n[3/4] 生成分类对比图…')
    plot_model_comparison(results, nb_result, os.path.join(FIGURE_DIR, 'fig_model_comparison.png'))

    print('\n[4/4] 生成情感分析图…')
    pos_means, neg_means = plot_sentiment_by_category(
        sentiment_df, class_names,
        os.path.join(FIGURE_DIR, 'fig_sentiment_by_category.png')
    )

    print('\n' + '=' * 60)
    print('综合可视化完成')
    print('=' * 60)
    print(f'  共生成图表:')
    for f in ['fig_model_comparison.png', 'fig_sentiment_by_category.png']:
        path = os.path.join(FIGURE_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f'    {f} ({size:.0f} KB)')


if __name__ == '__main__':
    main()
