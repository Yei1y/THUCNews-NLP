# 发现与决策

## 需求
- NLP 方向课程论文
- 以文本分类（LSTM）为主线
- 情感分析（词典型）作为对比
- 配合可视化分析
- LaTeX 编写，正文 ≤6 页
- 含简单引言，无摘要和文献综述

## 研究发现

### 数据集：THUCNews（Kaggle 版）
- 来源：Kaggle `xianhuizhang/thucnews`，已下载至本地（553MB zip）
- 10 个类别：finance, realty, stocks, education, science, society, politics, sports, game, entertainment
- 数据量：训练集 180,000 条 / 验证集 ~10,000 条 / 测试集 ~10,000 条
- 格式：`文本\t标签索引`，UTF-8 纯文本
- 附带预训练词向量（SougouNews + Tencent）和词汇表（vocab.pkl）
- 文本为原始中文（未分词），需用 jieba 分词

### 情感词典：大连理工情感词汇本体库（DUTIR）
- 7 大类 21 小类情感（乐、好、怒、哀、惧、恶、惊）
- 27,000+ 词汇，含情感强度和极性标注
- 可通过 `pip install cnsenti` 直接调用，无需手动下载
- 学术引用：徐琳宏等, 情报学报, 2008

### 技术方案设想
- 预处理：jieba 分词 → 去停用词
- 词向量：训练 Word2Vec（或在已有预训练向量上微调）
- 分类模型：LSTM（Embedding → LSTM → Dense）
- 调参目标：词向量维度、LSTM 单元数、dropout、学习率、batch size、epochs
- 情感分析：cnsenti（DUTIR 词典）统计正负情感词

## 技术决策
| 决策 | 理由 |
|------|------|
| THUCNews（10类） | 支持多分类任务，已下载就绪 |
| jieba 分词 + 自训练 Word2Vec | 灵活性高，便于调参实验 |
| LSTM 文本分类 | 用户指定，适合序列建模 |
| cnsenti（DUTIR 词典）做情感分析 | 学术权威，pip 安装无需手动下载 |
| 需考虑添加传统 ML 基线对比（如 Naive Bayes + TF-IDF） | 彰显 LSTM 的优势 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
|      |         |

## 资源
—

## 视觉/浏览器发现
<!-- 关键：每执行2次查看/浏览器操作后必须更新此部分 -->
<!-- 多模态内容必须立即以文本形式记录 -->
—

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
