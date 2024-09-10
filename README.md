# Financial_agent

## 简介

根据用户问题自主查询基金表数据库或检索公司招股书文档库的智能LLM系统

## 功能框架

- **意图识别**：基于ICL技术和提示词工程，将用户意图准确分为数据查询(NL2SQL任务)和文本检索(RAG任务)，同时对文本检索任务提取公司名称和关键词。

- **数据查询**：使用RAG-ICL技术，并引入反馈重试机制，通过SQL查询报错和COT技术构造提示词，提升数据查询任务的准确度，并对提示词进行调整以对齐用户问题中关于保留位数的要求。

- **文本理解**：设计基于BM25的两级检索机制，首先通过公司名称检索得到包含公司信息的文档名，然后使用关键词检索获取相关片段，最后构造提示词调用大模型生成回答。

![](http://www.linfeng-coding.top:85/i/2024/09/06/xvid69.png)

## 安装

请确保您已经安装了 Conda 和 Pip，然后按照以下步骤安装所需的依赖项：

- 安装 `FAISS-GPU`:
```bash
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0
```

- 安装 Python 依赖项:

```bash
pip install -r requirements.txt
```