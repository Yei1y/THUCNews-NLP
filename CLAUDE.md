# 大数据案例分析 — NLP 课程论文项目

## 项目概述
NLP 课程论文：以 THUCNews 数据集进行文本分类（LSTM 为主，Naive Bayes 为基线）和情感分析（DUTIR 词典），配合可视化对比，最终产出 LaTeX 论文（正文 ≤6 页）。

## 数据
- **THUCNews**（10 类）：finance, realty, stocks, education, science, society, politics, sports, game, entertainment
- 格式：`文本\t标签索引`，UTF-8
- 路径：`data/THUCNews/`
- 来源：Kaggle（已下载至项目根目录 `thucnews.zip`）

## Python 环境
- **conda env**: `python31111`（`D:\anaconda\envs\python31111`）
- **核心依赖**：pytorch, jieba, gensim, cnsenti, scikit-learn, matplotlib, seaborn, pandas, numpy

## 运行指南
```bash
conda activate python31111
# 或使用完整路径：
D:\anaconda\envs\python31111\python.exe script.py
```

## 脚本执行顺序
1. `01_data_prep.py` — 数据解压与预处理
2. `02_eda_viz.py` — 探索性数据分析与可视化
3. `03_word2vec.py` — 训练词向量
4. `04_lstm_classifier.py` — LSTM 分类（含调参） + 基线对比
5. `05_sentiment_analysis.py` — 情感分析（cnsenti）
6. `06_comparison_viz.py` — 综合对比可视化与论文用图

## 论文
- LaTeX 编写，正文 ≤6 页
- 含简单引言，无摘要，无文献综述
- 路径：`report/paper.tex`
- 编译：XeLaTeX（支持中文）

## 重要约定
- 所有结果（数值、图表）输出到 `output/` 目录
- 分析结论写在独立文档（如 reports 或 results.md），不在代码中 print 结论
- 设置随机种子保证可复现性
- 情感分析使用 `cnsenti`（内嵌大连理工情感词汇本体库 DUTIR）
