# 基于 LSTM 的中文文本分类与情感分析对比研究

NLP 课程论文项目：以 THUCNews 数据集进行文本分类（LSTM 为主，Naive Bayes 为基线）和情感分析（基于 DUTIR 词典），配合可视化对比，最终产出 LaTeX 论文。

## 项目概述

本项目完成了一篇 NLP 方向的课程论文，探究以下三方面问题：

1. **LSTM 在中文短文本分类中的有效性** — 在 THUCNews 数据集上达到 88.45% 测试准确率
2. **传统方法与深度方法的性能差距** — Naive Bayes + TF-IDF 基线仅 29.59%
3. **新闻类别的情感分布特征** — 体育/娱乐正面突出，社会/政治负面较多

## 项目结构

```
├── data/
│   ├── THUCNews/          # THUCNews 原始数据（训练/验证/测试集）
│   ├── processed/          # 预处理后的数据（词汇表、编码序列、情感结果）
│   └── stopwords.txt       # 停用词表
├── scripts/
│   ├── 01_data_prep.py         # 数据预处理：分词、去停用词、构建词汇表
│   ├── 02_eda_viz.py           # 探索性数据分析和可视化
│   ├── 03_word2vec.py          # Word2Vec 词向量训练
│   ├── 04_lstm_classifier.py   # LSTM 分类 + Naive Bayes 基线
│   ├── 05_sentiment_analysis.py # 基于 DUTIR 词典的情感分析
│   └── 06_comparison_viz.py    # 综合对比可视化
├── output/
│   ├── figures/           # 所有图表（共 11+ 张）
│   ├── models/            # 训练好的模型（Word2Vec, LSTM, embedding matrix）
│   └── tables/            # 结果汇总表
├── report/
│   └── paper.tex          # LaTeX 论文源文件
├── thucnews.zip           # THUCNews 原始压缩包
├── CLAUDE.md              # 项目配置文件
└── README.md
```

## 数据集

**THUCNews** — 清华大学发布的中文新闻分类数据集（10 类子集）：

| 类别 | 英文 | 类别 | 英文 |
|------|------|------|------|
| 财经 | finance | 科技 | science |
| 房产 | realty | 社会 | society |
| 股票 | stocks | 体育 | sports |
| 教育 | education | 游戏 | game |
| 政治 | politics | 娱乐 | entertainment |

- 样本量：训练集 180,000 / 验证集 10,000 / 测试集 10,000
- 文本类型：新闻标题（短文本，平均长度约 8 个词）
- 来源：Kaggle — [xianhuizhang/thucnews](https://www.kaggle.com/datasets/xianhuizhang/thucnews)

## 方法

### 文本分类
- **主模型**：LSTM + Word2Vec 预训练词向量
  - 词向量维度：200（Skip-gram 自训练）
  - 隐藏单元：64 / 128，Dropout：0.5
  - 最佳配置测试准确率：**88.45%**
- **基线模型**：Naive Bayes + TF-IDF
  - 测试准确率：29.59%

### 情感分析
- **方法**：基于词典的情感分析
- **词典**：大连理工情感词汇本体库（DUTIR），通过 `cnsenti` 库调用
- **结果**：测试集正面 43.3% / 中性 41.7% / 负面 15.0%

## 环境配置

```bash
conda activate python31111
pip install torch jieba gensim cnsenti scikit-learn matplotlib seaborn pandas numpy
```

## 运行顺序

```bash
python scripts/01_data_prep.py         # 数据预处理
python scripts/02_eda_viz.py           # 探索性数据分析
python scripts/03_word2vec.py          # 词向量训练
python scripts/04_lstm_classifier.py   # LSTM 分类 + NB 基线
python scripts/05_sentiment_analysis.py # 情感分析
python scripts/06_comparison_viz.py    # 综合对比可视化
```

## 论文

LaTeX 论文位于 `report/paper.tex`，使用 XeLaTeX 编译（支持中文），正文不超过 6 页。

## 结果摘要

| 模型 | 准确率 | F1 (macro) | 训练时间 |
|------|--------|------------|----------|
| LSTM (h=64, d=0.5) | **0.8845** | **0.8845** | 87.6s |
| LSTM (h=128, d=0.5) | 0.8790 | 0.8791 | 143.0s |
| Naive Bayes + TF-IDF | 0.2959 | 0.2937 | 1.9s |

## 许可

MIT License
