#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_eda_viz.py — 探索性数据分析与可视化
功能：文本长度分布、类别分布、高频词可视化
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
# 注册中文字体（Windows 路径）
for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 路径 ----------
PROCESSED_PATH = os.path.join('..', 'data', 'processed', 'processed.pkl')
FIGURE_DIR = os.path.join('..', 'output', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# 种子
random.seed(42)
np.random.seed(42)

# 配色 — 学术蓝橙
COLORS = ['#4DBBD5', '#F39B7F', '#00A087', '#91D1C2',
          '#8491B4', '#FCC5A1', '#E64B35', '#3C5488',
          '#7E6148', '#DC0000']


def set_style():
    """统一图表样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    })


def plot_text_length_distribution(tokenized_texts, class_names, labels, save_path):
    """文本长度分布图"""
    lengths = [len(tokens) for tokens in tokenized_texts]
    lengths_series = pd.Series(lengths)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：整体直方图
    ax = axes[0]
    ax.hist(lengths, bins=80, color=COLORS[0], edgecolor='white', alpha=0.85)
    median_len = lengths_series.median()
    mean_len = lengths_series.mean()
    ax.axvline(median_len, color=COLORS[1], ls='--', lw=1.5, label=f'中位数={median_len:.0f}')
    ax.axvline(mean_len, color=COLORS[2], ls='--', lw=1.5, label=f'均值={mean_len:.0f}')
    ax.set_xlabel('文本长度（词数）')
    ax.set_ylabel('频数')
    ax.set_title('文本长度分布')
    ax.legend()

    # 右：各类别平均长度
    ax = axes[1]
    df = pd.DataFrame({'length': lengths, 'label': labels})
    avg_len = df.groupby('label')['length'].mean()
    bars = ax.bar(range(len(avg_len)), avg_len.values, color=COLORS[:len(avg_len)])
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_xlabel('类别')
    ax.set_ylabel('平均长度（词数）')
    ax.set_title('各类别平均文本长度')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'  已保存: {save_path}')

    # 输出统计量到文件
    stats = {
        'count': len(lengths),
        'mean': round(mean_len, 2),
        'median': round(median_len, 2),
        'std': round(lengths_series.std(), 2),
        'min': int(lengths_series.min()),
        'max': int(lengths_series.max()),
        'q25': round(lengths_series.quantile(0.25), 2),
        'q75': round(lengths_series.quantile(0.75), 2),
    }
    return stats


def plot_category_distribution(labels, class_names, save_path):
    """类别分布饼图 + 条形图"""
    counts = pd.Series(labels).value_counts().sort_index()
    values = [counts[i] for i in range(len(class_names))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：条形图
    ax = axes[0]
    bars = ax.bar(class_names, values, color=COLORS)
    ax.set_xlabel('类别')
    ax.set_ylabel('样本数')
    ax.set_title('各类别样本分布')
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val}', ha='center', va='bottom', fontsize=7)

    # 右：饼图
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        values, labels=class_names, autopct='%1.1f%%',
        colors=COLORS, startangle=90, pctdistance=0.85
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax.set_title('类别占比')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'  已保存: {save_path}')

    return dict(zip(class_names, values))


def plot_top_words(train_tokenized, class_names, train_labels, n_top=20, save_path=None):
    """各类别及整体高频词条形图"""
    # 整体高频词
    from collections import Counter
    all_words = Counter()
    for words in train_tokenized:
        all_words.update(words)

    top_words = all_words.most_common(n_top)
    words, freqs = zip(*top_words)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(words)), freqs, color=COLORS[0])
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('词频')
    ax.set_title(f'全体 Top-{n_top} 高频词')
    for bar, val in zip(bars, freqs):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f'  已保存: {save_path}')
    else:
        plt.show()

    return top_words


def main():
    print('=' * 60)
    print('02 — 探索性数据分析与可视化')
    print('=' * 60)

    # 加载预处理数据
    print('\n[1/3] 加载预处理数据…')
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']
    print(f'  样本数: {len(data["train_encoded"])} (train)')
    print(f'  词汇表大小: {data["vocab_size"]}')

    set_style()

    # 1. 文本长度分布
    print('\n[2/3] 文本长度分布分析…')
    stats = plot_text_length_distribution(
        data['train_tokenized'], class_names, data['train_labels'],
        os.path.join(FIGURE_DIR, 'fig1_text_length_dist.png')
    )
    print(f'  长度统计: 均值={stats["mean"]}, 中位数={stats["median"]}, '
          f'std={stats["std"]}, 范围=[{stats["min"]}, {stats["max"]}]')

    # 2. 类别分布
    print('\n  类别分布…')
    dist = plot_category_distribution(
        data['train_labels'], class_names,
        os.path.join(FIGURE_DIR, 'fig2_category_dist.png')
    )

    # 3. 高频词
    print('\n[3/3] 高频词分析…')
    top_words = plot_top_words(
        data['train_tokenized'], class_names, data['train_labels'], n_top=20,
        save_path=os.path.join(FIGURE_DIR, 'fig3_top_words.png')
    )
    print(f'  Top-5 高频词: {[w for w, _ in top_words[:5]]}')

    print('\n' + '=' * 60)
    print('EDA 完成 — 图表已保存至 output/figures/')
    print('=' * 60)


if __name__ == '__main__':
    main()
