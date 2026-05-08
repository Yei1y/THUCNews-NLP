# 进度日志

## 会话：2026-05-08

### 阶段 1：需求与发现
- **状态：** complete
- **开始时间：** 2026-05-08 18:21
- 执行的操作：
  - 创建项目目录
  - 创建 task_plan.md、findings.md、progress.md
  - 搜索并确认 THUCNews 数据集信息
  - 搜索并确认中文情感词典（DUTIR via cnsenti）
  - 下载 THUCNews（Kaggle，553MB）
- 创建/修改的文件：
  - task_plan.md
  - findings.md
  - progress.md

### 阶段 2：规划与结构
- **状态：** complete
- 执行的操作：
  - 创建项目子目录（data, scripts, output, report）
  - 创建项目级 CLAUDE.md
  - 更新 task_plan.md 决策表
- 创建/修改的文件：
  - CLAUDE.md
  - task_plan.md (更新)

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 01 预处理 | 20万条原始数据 | 分词+词汇表+编码保存 | 训练集18万/验证1万/测试1万，词汇表30601 | ✅ |
| 02 EDA | 预处理数据 | 生成3张图表 | fig1~3 全部生成 | ✅ |
| 03 Word2Vec | 分词文本 | 训练词向量模型 | 30599词/200维，55s，覆盖率100% | ✅ |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-05-08 | 01_data_prep.py: load_data 分割列数错误 | 1 | 数据格式为 text\tlabel（2列），原代码误判为3列 |
| 2026-05-08 | 02_eda_viz.py: matplotlib 中文字体缺失 | 1 | 注册 SimHei 字体 + 清除 matplotlib 缓存 |
| 2026-05-08 | PyTorch 未安装 | 1 | pip install torch --index-url https://download.pytorch.org/whl/cpu （被中断待重试） |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 5 —— 文本分类建模（正在安装 PyTorch） |
| 我要去哪里？ | 完成 LSTM 分类 → 情感分析 → 对比可视化 → 论文 |
| 目标是什么？ | 完成 NLP 课程论文：文本分类+情感分析+可视化，LaTeX 输出 |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 见上方及 progress.md 当前会话记录 |

---
*每个阶段完成后或遇到错误时更新此文件*

## 会话：2026-05-08 编码阶段

### 阶段 3-7：编码与运行
- **状态：** complete
- **执行的操作：**
  - 编写 01_data_prep.py：jieba 分词 + 去停用词 + 构建词汇表 + 编码（30601 词）
  - 编写 02_eda_viz.py：文本长度/类别分布/高频词 3 张 EDA 图
  - 编写 03_word2vec.py：训练 Skip-gram Word2Vec（200维，55s，100%覆盖）
  - 编写 04_lstm_classifier.py：
    - LSTM 调参（h=64, 128；d=0.5），最佳 test acc=0.8845
    - Naive Bayes + TF-IDF 基线 test acc=0.2959
    - 生成混淆矩阵、训练曲线
  - 编写 05_sentiment_analysis.py：cnsenti（DUTIR）情感分析，测试集 10000 条
  - 编写 06_comparison_viz.py：生成分类对比 + 情感对比论文用图
- **修改的文件：**
  - scripts/01_data_prep.py ~ 06_comparison_viz.py
  - output/figures/（12 张图）
  - output/tables/classification_results.md
  - data/processed/sentiment_results.pkl
  - output/models/（word2vec.model, lstm_best.pt 等）
- **遇到的问题：**
  - 数据格式误判（2列≠3列）→ 修复 load_data
  - matplotlib 中文显示 → 注册 SimHei
  - PyTorch CPU 训练过慢 → 优化模型（单层 LSTM + 平均池化 + 截断50 + 采样3K/类）
  - LSTM 卡在随机猜测 → 改用 mean pooling 替代 last hidden state
