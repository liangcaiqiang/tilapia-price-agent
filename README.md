# 罗非鱼鱼价预测 Agent

一个面向罗非鱼市场研究的 Python Agent，用于整合历史价格、供给、库存、饲料销量、出口与新闻情报等信息，辅助进行鱼价趋势分析和预测。

## 主要功能

- 罗非鱼价格预测与特征工程
- 自动生成模拟/历史市场数据并训练预测模型
- 新闻与网页情报采集
- 使用 LLM 从新闻文本中提取市场指标
- 周度公开数据采集与整理
- Firecrawl 补缺研究
- 可视化市场指标与预测结果
- 一键启动交互式 Agent

## 项目结构

- `run_agent.py`：主入口，整合预测模型与新闻情报模块
- `tilapia_price_agent.py`：价格预测、特征工程和可视化核心
- `news_intelligence.py`：新闻采集、LLM 提取和数据存储
- `weekly_data_collector.py`：周度公开数据采集
- `firecrawl_gap_research.py`：Firecrawl 数据补缺研究
- `firecrawl_missing_data_20260803.py`：特定缺失数据补充脚本
- `config.example.json`：配置模板
- `tilapia_data/`：已采集的市场研究数据与历史结果

## 安装

建议使用 Python 3.12+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## 配置

复制配置模板：

```bash
cp config.example.json config.json
```

然后在本地 `config.json` 中填写自己的 LLM API Key。

> `config.json` 已加入 `.gitignore`，不会提交到 GitHub。

如需使用 Firecrawl，可以设置 `FIRECRAWL_API_KEY` 环境变量，并把值替换为你自己的密钥；不要把真实密钥提交到仓库。

## 使用

标准启动：

```bash
python3 run_agent.py
```

启动并启用新闻情报：

```bash
python3 run_agent.py --with-news
```

使用自己的历史数据：

```bash
python3 run_agent.py --data your_data.csv
```

周度数据采集：

```bash
python3 weekly_data_collector.py
```

## 安全说明

真实 API Key、个人配置、虚拟环境、缓存和本地凭据文件不会提交到仓库。请不要把真实密钥直接写进源代码或 `config.example.json`。

## 数据说明

`tilapia_data/` 中包含项目运行过程中采集或整理的研究数据。不同数据源可能有各自的使用条款，使用时请核对原始来源和授权范围。
