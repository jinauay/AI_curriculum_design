\# 可解释谣言检测系统



课程：人工智能导论大作业2026



\---

\## 1.项目简介

本项目实现了一个可解释谣言检测系统（Explainable Rumor Detection System），用于检测社交媒体推文是否为谣言，并生成可理解的解释文本。



\## 2.系统整体流程

推文

&#x20; ↓

文本预处理

&#x20; ↓

BERTweet 分类器

&#x20; ↓

预测标签 + 置信度

&#x20; ↓

SJTU DeepSeek 大模型

&#x20; ↓

生成解释文本



\## 3.项目结构

Rumor\_Detection/

├── README.md

├── requirements.txt

├── .env.example

├── data/

│   ├── train.csv

│   └── val.csv

├── checkpoints/

│   └── best\_model/

└── src/

&#x20;   ├── preprocess.py

&#x20;   ├── dataset.py

&#x20;   ├── model.py

&#x20;   ├── train.py

&#x20;   ├── evaluate.py

&#x20;   ├── infer.py

&#x20;   └── explain.py



\## 4.环境配置

Python 3.10+，PyTorch 2.x



\## 5.API配置

在交我办申请，获得api-key



\## 5.使用数据

数据集包含 “train.csv” 和 “val.csv”

数据量训练集约 3000 条，验证集约 500 条



\## 6.模型结构

分类器：`vinai/bertweet-base`

输出：二分类

损失函数：CrossEntropyLoss

解释模块：SJTU DeepSeek 大模型



\## 7.模型训练

保存在checkpoints/best\_model/bertweet.pth



\## 8.项目亮点

使用 BERTweet 微调分类器保证分类精度

调用 SJTU 大模型生成可解释文本



