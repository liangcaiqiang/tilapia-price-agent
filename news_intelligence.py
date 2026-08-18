#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════
  罗非鱼新闻情报模块  news_intelligence.py
  Tilapia News Intelligence Module
══════════════════════════════════════════════════════════════════

功能流程：
  1. 从多个渠道采集罗非鱼相关新闻（搜索引擎/行业网站/手动输入URL）
  2. 调用大模型API（OpenAI / Claude / DeepSeek），从新闻中智能提取5个核心指标
  3. 将提取结果结构化存储（JSON日志 + CSV汇总）
  4. 与预测Agent无缝对接，自动更新预测模型

使用前配置（config.json）：
{
    "llm": {
        "provider": "openai",           // "openai" | "claude" | "deepseek"
        "api_key":  "sk-xxxxxxxx",
        "model":    "gpt-4o-mini",      // 或 "claude-sonnet-4-20250514" | "deepseek-chat"
        "base_url": null                 // DeepSeek需设为 "https://api.deepseek.com"
    },
    "news_sources": {
        "search_enabled": true,
        "search_keywords": ["罗非鱼 价格", "罗非鱼 行情", "tilapia price"],
        "industry_sites": [
            "https://www.shuichan.cc",
            "https://www.fishfirst.cn"
        ]
    },
    "data_dir": "./tilapia_data"
}

依赖：
  pip install requests  (通常已预装)
  无其他第三方依赖，所有LLM调用均通过HTTP API完成
══════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin
from typing import List, Dict, Optional, Any
import urllib.request
import urllib.error
import ssl

# ════════════════════════════════════════════════════════════
# 1. 配置管理
# ════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",
        "api_key": "",
        "model": "gpt-4o-mini",
        "base_url": None,
    },
    "news_sources": {
        "search_enabled": True,
        "search_keywords": [
            "罗非鱼 价格 行情",
            "罗非鱼 出塘价",
            "罗非鱼 存塘量",
            "罗非鱼 放苗",
            "罗非鱼 饲料 销量",
            "罗非鱼 加工厂 库存",
            "tilapia price export",
            "罗非鱼 美国 出口",
        ],
        "industry_sites": [],
    },
    "data_dir": "./tilapia_data",
}


class Config:
    """配置管理器"""

    def __init__(self, config_path: str = "config.json"):
        self.path = config_path
        self.data = DEFAULT_CONFIG.copy()
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_cfg = json.load(f)
            self._deep_merge(self.data, user_cfg)
        else:
            print(f"[配置] 未找到 {config_path}，使用默认配置")
            print(f"[配置] 请创建 config.json 并填写 API Key 后再运行")

    def _deep_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @property
    def llm(self): return self.data["llm"]

    @property
    def news(self): return self.data["news_sources"]

    @property
    def data_dir(self): return self.data["data_dir"]


# ════════════════════════════════════════════════════════════
# 2. 统一LLM调用接口（支持 OpenAI / Claude / DeepSeek）
# ════════════════════════════════════════════════════════════

class LLMClient:
    """
    统一的大模型调用接口。
    通过 HTTP API 调用，无需安装 SDK。

    支持：
      - OpenAI (GPT-4o, GPT-4o-mini, ...)
      - Anthropic Claude (claude-sonnet-4-20250514, ...)
      - DeepSeek (deepseek-chat, deepseek-reasoner)
      - Google Gemini (gemini-2.0-flash, gemini-2.5-pro, ...)
      - 任何兼容 OpenAI 格式的 API
    """

    # 各厂商的默认 Base URL
    PROVIDER_URLS = {
        "openai":   "https://api.openai.com/v1",
        "claude":   "https://api.anthropic.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai",
    }

    def __init__(self, provider: str, api_key: str,
                 model: str = None, base_url: str = None):
        self.provider = provider.lower()
        self.api_key  = api_key
        self.model    = model or self._default_model()
        self.base_url = (base_url or
                         self.PROVIDER_URLS.get(self.provider,
                                                "https://api.openai.com/v1"))

        if not self.api_key:
            raise ValueError(
                f"[LLM] API Key 未配置！请在 config.json 中设置 llm.api_key\n"
                f"  当前 provider: {self.provider}\n"
                f"  需要的 Key 格式: {'sk-...' if 'openai' in self.provider else 'key...'}"
            )
        # SSL上下文（兼容某些环境）
        self._ssl_ctx = ssl.create_default_context()

    def _default_model(self):
        return {
            "openai":   "gpt-4o-mini",
            "claude":   "claude-sonnet-4-20250514",
            "deepseek": "deepseek-chat",
            "gemini":   "gemini-2.0-flash",
        }.get(self.provider, "gpt-4o-mini")

    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """
        统一聊天接口，屏蔽各厂商差异。

        Returns: LLM响应的文本内容
        """
        if self.provider == "claude":
            return self._call_claude(system_prompt, user_message,
                                     temperature, max_tokens)
        else:
            # OpenAI / DeepSeek / Gemini / 兼容格式
            return self._call_openai_compatible(system_prompt, user_message,
                                                 temperature, max_tokens)

    def _call_openai_compatible(self, system: str, user: str,
                                 temp: float, max_tok: int) -> str:
        url  = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temp,
            "max_tokens": max_tok,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(body).encode('utf-8')
        req  = urllib.request.Request(url, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(
                f"[LLM] API调用失败 ({e.code}): {err_body[:300]}")
        except Exception as e:
            raise RuntimeError(f"[LLM] 调用异常: {e}")

    def _call_claude(self, system: str, user: str,
                     temp: float, max_tok: int) -> str:
        url  = f"{self.base_url}/messages"
        body = {
            "model": self.model,
            "max_tokens": max_tok,
            "system": system,
            "messages": [
                {"role": "user", "content": user},
            ],
            "temperature": temp,
        }
        headers = {
            "Content-Type":    "application/json",
            "x-api-key":       self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = json.dumps(body).encode('utf-8')
        req  = urllib.request.Request(url, data=data, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=90) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            # Claude 返回格式: content[0].text
            return result["content"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(
                f"[LLM/Claude] API调用失败 ({e.code}): {err_body[:300]}")
        except Exception as e:
            raise RuntimeError(f"[LLM/Claude] 调用异常: {e}")

    def test_connection(self) -> bool:
        """测试API连通性"""
        try:
            resp = self.chat("你是一个助手。", "请回复'OK'两个字母。",
                             temperature=0, max_tokens=10)
            return 'OK' in resp.upper()
        except Exception as e:
            print(f"[LLM] 连接测试失败: {e}")
            return False


# ════════════════════════════════════════════════════════════
# 3. 新闻采集器
# ════════════════════════════════════════════════════════════

class NewsCollector:
    """
    多渠道新闻采集：
      - 搜索引擎（通过搜索结果页解析）
      - 指定URL直接抓取
      - 手动输入新闻文本
    """

    # 常用的搜索引擎搜索URL模板
    SEARCH_TEMPLATES = {
        "bing": "https://www.bing.com/search?q={query}&setlang=zh-CN",
        "sogou": "https://www.sogou.com/web?query={query}",
    }

    def __init__(self):
        self._ssl_ctx = ssl.create_default_context()
        self._headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def fetch_url(self, url: str, timeout: int = 15) -> Optional[str]:
        """抓取URL内容，返回纯文本（去除HTML标签）"""
        try:
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, context=self._ssl_ctx,
                                        timeout=timeout) as resp:
                raw = resp.read()
                # 尝试检测编码
                encoding = 'utf-8'
                ct = resp.headers.get('Content-Type', '')
                m = re.search(r'charset=([^\s;]+)', ct)
                if m:
                    encoding = m.group(1)
                html = raw.decode(encoding, errors='replace')
            return self._html_to_text(html)
        except Exception as e:
            print(f"  [采集] 抓取失败 {url[:60]}... : {e}")
            return None

    def search_news(self, keywords: List[str],
                    engine: str = "bing",
                    max_results: int = 5) -> List[Dict]:
        """
        通过搜索引擎搜索新闻。
        返回: [{"title": ..., "url": ..., "snippet": ...}, ...]
        """
        results = []
        template = self.SEARCH_TEMPLATES.get(engine, self.SEARCH_TEMPLATES["bing"])

        for kw in keywords[:4]:  # 限制搜索轮次
            url = template.format(query=quote_plus(kw))
            html_text = self.fetch_url(url, timeout=10)
            if not html_text:
                continue
            # 从搜索结果中提取URL（简单正则）
            urls_found = re.findall(
                r'https?://[^\s<>"\']+(?:shuichan|fishfirst|cnfisheries|'
                r'thepaper|163\.com|sohu\.com|sina\.com|qq\.com|'
                r'chyxx|huaon|seafood)[^\s<>"\']*',
                html_text
            )
            for u in urls_found[:max_results]:
                u = u.rstrip('.,;)')
                if u not in [r['url'] for r in results]:
                    results.append({
                        "title": "",
                        "url": u,
                        "snippet": "",
                        "keyword": kw,
                    })

            if len(results) >= max_results:
                break
            time.sleep(1)  # 礼貌间隔

        return results[:max_results]

    def fetch_article(self, url: str) -> Optional[Dict]:
        """抓取单篇文章，返回结构化内容"""
        text = self.fetch_url(url)
        if not text or len(text) < 100:
            return None
        # 截取合理长度（LLM上下文限制）
        text = text[:8000]
        return {
            "url": url,
            "text": text,
            "fetched_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _html_to_text(html: str) -> str:
        """简单的HTML转纯文本"""
        # 移除script和style
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL|re.I)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL|re.I)
        html = re.sub(r'<[^>]+>', ' ', html)
        # 清理空白
        html = re.sub(r'\s+', ' ', html)
        html = re.sub(r'&nbsp;', ' ', html)
        html = re.sub(r'&[a-z]+;', '', html)
        return html.strip()

    @staticmethod
    def from_text(text: str, source: str = "manual_input") -> Dict:
        """手动输入新闻文本"""
        return {
            "url": source,
            "text": text[:8000],
            "fetched_at": datetime.now().isoformat(),
        }


# ════════════════════════════════════════════════════════════
# 4. LLM 指标提取器
# ════════════════════════════════════════════════════════════

# 核心提示词：引导LLM从新闻中提取5个维度的结构化数据
EXTRACTION_SYSTEM_PROMPT = """你是一个专业的水产行业数据分析师，专门分析罗非鱼市场。
你需要从新闻报道中提取以下5个核心市场指标的信息。

【5个核心指标及其含义】

1. **美国收购商仓库库存** (us_inventory)
   - 含义：美国罗非鱼进口商/收购商的库存水平
   - 判断线索：美国进口量变化、美国库存积压/消化情况、采购订单增减
   - 量化方式：以100为基准，>100表示库存偏高（压价），<100表示偏低（支撑价格）

2. **中国加工厂仓库库存** (cn_factory_inventory)
   - 含义：国内罗非鱼加工厂的冻品/原料库存
   - 判断线索：加工厂开工率、收购积极性、冻品库存消化速度
   - 量化方式：以100为基准，>100表示库存偏高

3. **鱼塘存塘量** (pond_stock_level)
   - 含义：养殖户鱼塘中罗非鱼的存量
   - 判断线索：存塘量大小、养殖密度、压塘情况、出塘速度
   - 量化方式：以100为基准，>100表示存塘偏多（供应压力）

4. **苗种放苗量** (fry_release_volume)
   - 含义：当期苗种投放规模（4个月后影响供应量）
   - 判断线索：苗种需求、鱼苗价格、养殖户投苗积极性
   - 量化方式：以100为基准，>100表示投苗规模偏大

5. **饲料厂饲料销量** (feed_sales_volume)
   - 含义：主流饲料厂罗非鱼饲料的销售量
   - 判断线索：饲料出货量、饲料价格变动、养殖户补料频率
   - 量化方式：以100为基准，>100表示销量偏高

【其他有价值的信息（如果新闻提到请一并提取）】
- 出塘价格（元/公斤）
- 饲料成本变动
- 天气风险事件
- 出口市场变化
- 政策变动

【输出格式要求】
请严格以JSON格式返回，不要有其他文字：
```json
{
    "extracted_date": "新闻报道的大致时间，格式 YYYY-MM",
    "confidence": "high/medium/low - 你对提取结果的置信度",
    "indicators": {
        "us_inventory": {"value": 数字或null, "evidence": "原文依据", "direction": "up/down/stable/unknown"},
        "cn_factory_inventory": {"value": 数字或null, "evidence": "原文依据", "direction": "up/down/stable/unknown"},
        "pond_stock_level": {"value": 数字或null, "evidence": "原文依据", "direction": "up/down/stable/unknown"},
        "fry_release_volume": {"value": 数字或null, "evidence": "原文依据", "direction": "up/down/stable/unknown"},
        "feed_sales_volume": {"value": 数字或null, "evidence": "原文依据", "direction": "up/down/stable/unknown"}
    },
    "price_info": {
        "current_price": 数字或null,
        "price_unit": "元/公斤 或 元/斤",
        "price_trend": "up/down/stable/unknown"
    },
    "risk_events": ["风险事件描述，如台风、寒潮、疫病等"],
    "market_summary": "一句话市场总结",
    "data_quality_notes": "数据质量说明，如'新闻内容模糊，仅能推断方向'"
}
```

【重要注意事项】
- 如果新闻中没有提到某个指标，value设为null，direction设为"unknown"
- 不要编造数据！只根据新闻实际内容提取
- evidence 字段要引用新闻中的原文片段作为依据
- 如果只能判断方向（涨/跌）但无法量化，value设为null，但direction要填写
- 价格单位注意区分"元/公斤"和"元/斤"
"""

EXTRACTION_USER_TEMPLATE = """请分析以下罗非鱼行业新闻，提取5个核心市场指标：

【新闻来源】{source}
【新闻内容】
{content}

请按照要求的JSON格式返回提取结果。"""

# 多条新闻汇总提示词
AGGREGATION_SYSTEM_PROMPT = """你是一个水产行业数据分析师。
现在你有多条新闻的提取结果，请将这些数据综合分析，生成一个月度数据汇总。

【汇总规则】
1. 对同一指标如有多个来源，取加权平均（置信度high的权重3，medium权重2，low权重1）
2. 如果只有方向信息没有具体数值，根据方向推断：
   - up → 基准值 * 1.05~1.15
   - down → 基准值 * 0.85~0.95
   - stable → 基准值 * 0.98~1.02
3. 结合季节性常识校验数据合理性

请严格以JSON格式返回：
```json
{
    "month": "YYYY-MM",
    "us_inventory": 数字,
    "cn_factory_inventory": 数字,
    "pond_stock_level": 数字,
    "fry_release_volume": 数字,
    "feed_sales_volume": 数字,
    "price": 数字或null,
    "weather_risk": 0或1或2,
    "feed_cost_index": 数字,
    "export_index": 数字,
    "confidence": "high/medium/low",
    "sources_count": 数字,
    "analysis_summary": "综合分析说明"
}
```"""


class NewsExtractor:
    """
    利用LLM从新闻中提取结构化指标数据。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_from_article(self, article: Dict) -> Optional[Dict]:
        """从单篇文章中提取指标"""
        if not article or not article.get('text'):
            return None

        user_msg = EXTRACTION_USER_TEMPLATE.format(
            source=article.get('url', 'unknown'),
            content=article['text'][:6000],  # 限制长度
        )

        try:
            response = self.llm.chat(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=2000,
            )
            # 从响应中提取JSON
            parsed = self._parse_json_response(response)
            if parsed:
                parsed['source_url'] = article.get('url', '')
                parsed['extracted_at'] = datetime.now().isoformat()
            return parsed
        except Exception as e:
            print(f"  [提取] LLM调用失败: {e}")
            return None

    def aggregate_extractions(self, extractions: List[Dict],
                              target_month: str = None) -> Optional[Dict]:
        """
        多条提取结果汇总为月度数据。

        extractions: extract_from_article() 返回结果的列表
        target_month: 目标月份 "YYYY-MM"，默认取提取结果中的最新月份
        """
        if not extractions:
            return None

        if not target_month:
            target_month = datetime.now().strftime("%Y-%m")

        summary_text = json.dumps(extractions, ensure_ascii=False, indent=2)
        # 截断过长内容
        if len(summary_text) > 12000:
            summary_text = summary_text[:12000] + "\n... (截断)"

        user_msg = f"""以下是从 {len(extractions)} 条罗非鱼行业新闻中提取的数据：

{summary_text}

请汇总为 {target_month} 月度数据。"""

        try:
            response = self.llm.chat(
                system_prompt=AGGREGATION_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.15,
                max_tokens=1500,
            )
            result = self._parse_json_response(response)
            if result:
                result['month'] = target_month
                result['raw_extractions'] = len(extractions)
            return result
        except Exception as e:
            print(f"  [汇总] LLM调用失败: {e}")
            return None

    @staticmethod
    def _parse_json_response(text: str) -> Optional[Dict]:
        """从LLM响应中稳健提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 尝试从 ```json ... ``` 中提取
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except:
                pass

        # 尝试找到 { ... } 块
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass

        print(f"  [解析] 无法从LLM响应中提取JSON：{text[:200]}...")
        return None


# ════════════════════════════════════════════════════════════
# 5. 数据存储管理
# ════════════════════════════════════════════════════════════

class DataStore:
    """
    数据持久化存储：
      - tilapia_data/news_log/     原始新闻JSON日志（可溯源）
      - tilapia_data/extractions/  每次提取结果的JSON
      - tilapia_data/monthly.csv   月度汇总CSV（直接用于预测模型）
    """

    def __init__(self, data_dir: str = "./tilapia_data"):
        self.data_dir   = data_dir
        self.news_dir   = os.path.join(data_dir, "news_log")
        self.extract_dir = os.path.join(data_dir, "extractions")
        self.csv_path   = os.path.join(data_dir, "monthly.csv")

        for d in [self.data_dir, self.news_dir, self.extract_dir]:
            os.makedirs(d, exist_ok=True)

    def save_news(self, article: Dict) -> str:
        """保存原始新闻"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_hash = hashlib.md5(article.get('url', ts).encode()).hexdigest()[:8]
        filename = f"{ts}_{url_hash}.json"
        path = os.path.join(self.news_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        return path

    def save_extraction(self, extraction: Dict) -> str:
        """保存提取结果"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extract_{ts}.json"
        path = os.path.join(self.extract_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(extraction, f, ensure_ascii=False, indent=2)
        return path

    def update_monthly_csv(self, monthly_data: Dict):
        """将月度汇总数据追加/更新到CSV"""
        import pandas as pd

        month = monthly_data.get('month', datetime.now().strftime('%Y-%m'))
        row = {
            'date':                  f"{month}-01",
            'price':                 monthly_data.get('price', ''),
            'us_inventory':          monthly_data.get('us_inventory', 100),
            'cn_factory_inventory':  monthly_data.get('cn_factory_inventory', 100),
            'pond_stock_level':      monthly_data.get('pond_stock_level', 100),
            'fry_release_volume':    monthly_data.get('fry_release_volume', 100),
            'feed_sales_volume':     monthly_data.get('feed_sales_volume', 100),
            'feed_cost_index':       monthly_data.get('feed_cost_index', 100),
            'weather_risk':          monthly_data.get('weather_risk', 0),
            'export_index':          monthly_data.get('export_index', 100),
            'confidence':            monthly_data.get('confidence', 'low'),
            'sources_count':         monthly_data.get('sources_count', 0),
            'updated_at':            datetime.now().isoformat(),
        }

        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            # 更新已有月份或追加
            if month + "-01" in df['date'].values:
                idx = df[df['date'] == month + "-01"].index[0]
                for k, v in row.items():
                    df.at[idx, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.sort_values('date', inplace=True)
        df.to_csv(self.csv_path, index=False, encoding='utf-8')
        print(f"  [存储] 月度数据已更新: {self.csv_path}")
        return self.csv_path

    def get_recent_extractions(self, days: int = 30) -> List[Dict]:
        """获取最近N天的提取结果"""
        results = []
        cutoff = datetime.now() - timedelta(days=days)
        for f in sorted(os.listdir(self.extract_dir)):
            if not f.endswith('.json'):
                continue
            path = os.path.join(self.extract_dir, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if data.get('extracted_at'):
                    et = datetime.fromisoformat(data['extracted_at'])
                    if et >= cutoff:
                        results.append(data)
            except:
                pass
        return results


# ════════════════════════════════════════════════════════════
# 6. 新闻情报流水线（主管道）
# ════════════════════════════════════════════════════════════

class NewsIntelligencePipeline:
    """
    完整的新闻情报流水线：
      采集新闻 → LLM提取指标 → 数据存储 → 汇总月度数据

    一键式调用：
      pipeline.run_full_cycle()
    """

    def __init__(self, config: Config = None, config_path: str = "config.json"):
        self.config  = config or Config(config_path)
        self.store   = DataStore(self.config.data_dir)
        self.collector = NewsCollector()

        # 初始化LLM客户端
        llm_cfg = self.config.llm
        self.llm = LLMClient(
            provider=llm_cfg["provider"],
            api_key=llm_cfg["api_key"],
            model=llm_cfg.get("model"),
            base_url=llm_cfg.get("base_url"),
        )
        self.extractor = NewsExtractor(self.llm)

    def run_full_cycle(self, target_month: str = None,
                       extra_urls: List[str] = None,
                       extra_texts: List[str] = None) -> Optional[Dict]:
        """
        执行完整的数据采集-提取-汇总-存储流程。

        Args:
            target_month: 目标月份 "YYYY-MM"
            extra_urls:   额外指定的新闻URL列表
            extra_texts:  额外的新闻文本内容列表

        Returns: 月度汇总数据 dict
        """
        if not target_month:
            target_month = datetime.now().strftime("%Y-%m")

        print(f"\n{'═'*60}")
        print(f"  🔍 新闻情报采集 —— 目标月份：{target_month}")
        print(f"{'═'*60}")

        # ── Step 1: 采集新闻 ──
        print("\n[Step 1/4] 采集新闻...")
        articles = []

        # 搜索引擎
        if self.config.news.get("search_enabled"):
            keywords = self.config.news.get("search_keywords", [])
            print(f"  搜索关键词：{keywords[:4]}")
            search_results = self.collector.search_news(keywords, max_results=5)
            print(f"  搜索到 {len(search_results)} 条结果")
            for sr in search_results:
                art = self.collector.fetch_article(sr['url'])
                if art:
                    articles.append(art)
                    self.store.save_news(art)
                    print(f"  ✓ 已采集: {sr['url'][:60]}...")

        # 额外URL
        if extra_urls:
            for url in extra_urls:
                art = self.collector.fetch_article(url)
                if art:
                    articles.append(art)
                    self.store.save_news(art)
                    print(f"  ✓ 已采集URL: {url[:60]}...")

        # 额外文本
        if extra_texts:
            for txt in extra_texts:
                art = NewsCollector.from_text(txt)
                articles.append(art)
                self.store.save_news(art)
                print(f"  ✓ 已录入手动文本 ({len(txt)} 字)")

        print(f"\n  共采集 {len(articles)} 篇有效新闻")

        if not articles:
            print("  ⚠ 未采集到任何新闻，无法提取数据")
            return None

        # ── Step 2: LLM提取 ──
        print(f"\n[Step 2/4] LLM指标提取（{self.config.llm['provider']} / {self.config.llm['model']}）...")
        extractions = []
        for i, art in enumerate(articles):
            print(f"  提取 [{i+1}/{len(articles)}] {art['url'][:50]}...")
            ext = self.extractor.extract_from_article(art)
            if ext:
                extractions.append(ext)
                self.store.save_extraction(ext)
                conf = ext.get('confidence', '?')
                summary = ext.get('market_summary', '')[:40]
                print(f"    → 置信度: {conf}  摘要: {summary}...")
            time.sleep(0.5)  # 避免API限流

        print(f"\n  成功提取 {len(extractions)}/{len(articles)} 篇")

        if not extractions:
            print("  ⚠ 所有提取均失败")
            return None

        # ── Step 3: 汇总 ──
        print(f"\n[Step 3/4] 汇总为月度数据...")
        monthly = self.extractor.aggregate_extractions(extractions, target_month)
        if monthly:
            print(f"  ✓ 汇总完成（置信度: {monthly.get('confidence', '?')}）")
        else:
            print("  ⚠ 汇总失败，使用单条最佳提取结果")
            # 回退：取置信度最高的单条
            best = max(extractions,
                       key=lambda x: {'high':3,'medium':2,'low':1}.get(
                           x.get('confidence','low'), 0))
            monthly = self._single_to_monthly(best, target_month)

        # ── Step 4: 存储 ──
        print(f"\n[Step 4/4] 存储月度数据...")
        self.store.update_monthly_csv(monthly)

        # 打印结果
        self._print_summary(monthly)
        return monthly

    def run_from_urls(self, urls: List[str],
                      target_month: str = None) -> Optional[Dict]:
        """仅从指定URL列表采集+提取"""
        return self.run_full_cycle(
            target_month=target_month,
            extra_urls=urls,
        )

    def run_from_text(self, texts: List[str],
                      target_month: str = None) -> Optional[Dict]:
        """从手动输入的新闻文本提取"""
        cfg_backup = self.config.news.get("search_enabled", True)
        self.config.news["search_enabled"] = False
        result = self.run_full_cycle(
            target_month=target_month,
            extra_texts=texts,
        )
        self.config.news["search_enabled"] = cfg_backup
        return result

    def _single_to_monthly(self, extraction: Dict,
                           target_month: str) -> Dict:
        """将单条提取结果转为月度格式"""
        inds = extraction.get('indicators', {})
        price_info = extraction.get('price_info', {})
        return {
            'month': target_month,
            'us_inventory':         inds.get('us_inventory', {}).get('value') or 100,
            'cn_factory_inventory': inds.get('cn_factory_inventory', {}).get('value') or 100,
            'pond_stock_level':     inds.get('pond_stock_level', {}).get('value') or 100,
            'fry_release_volume':   inds.get('fry_release_volume', {}).get('value') or 100,
            'feed_sales_volume':    inds.get('feed_sales_volume', {}).get('value') or 100,
            'price':                price_info.get('current_price'),
            'weather_risk':         1 if extraction.get('risk_events') else 0,
            'feed_cost_index':      100,
            'export_index':         100,
            'confidence':           extraction.get('confidence', 'low'),
            'sources_count':        1,
        }

    @staticmethod
    def _print_summary(monthly: Dict):
        """打印月度汇总结果"""
        print(f"""
{'─'*60}
  📊 月度数据汇总 —— {monthly.get('month', '?')}
{'─'*60}
  ★ 美国仓库库存指数:   {monthly.get('us_inventory', '?')}
  ★ 加工厂仓库库存指数: {monthly.get('cn_factory_inventory', '?')}
  ★ 鱼塘存塘量指数:     {monthly.get('pond_stock_level', '?')}
  ★ 苗种放苗量指数:     {monthly.get('fry_release_volume', '?')}
  ★ 饲料厂饲料销量指数: {monthly.get('feed_sales_volume', '?')}
  ──────────────────
  出塘价格:            {monthly.get('price', '未知')} 元/kg
  天气风险:            {monthly.get('weather_risk', 0)}
  饲料成本指数:        {monthly.get('feed_cost_index', '?')}
  出口指数:            {monthly.get('export_index', '?')}
  ──────────────────
  数据置信度:          {monthly.get('confidence', '?')}
  新闻来源数:          {monthly.get('sources_count', '?')}
  综合分析:            {monthly.get('analysis_summary', '')}
{'─'*60}
""")


# ════════════════════════════════════════════════════════════
# 7. CLI 交互入口
# ════════════════════════════════════════════════════════════

def main():
    """命令行交互入口"""
    import sys

    print("""
╔═══════════════════════════════════════════════════════════════╗
║       🐟 罗非鱼新闻情报采集系统                                ║
║       News Intelligence for Tilapia Price Prediction          ║
╠═══════════════════════════════════════════════════════════════╣
║  用法:                                                        ║
║    python3 news_intelligence.py                    交互模式    ║
║    python3 news_intelligence.py --auto             自动采集    ║
║    python3 news_intelligence.py --url URL1 URL2    指定URL     ║
║    python3 news_intelligence.py --test             测试API     ║
╚═══════════════════════════════════════════════════════════════╝
""")

    config_path = "config.json"
    if not os.path.exists(config_path):
        print("  ⚠  首次运行，需要配置 config.json")
        print("  正在生成模板配置文件...")
        cfg = Config(config_path)
        cfg.save()
        print(f"  ✓ 已生成 {config_path}，请编辑并填写 API Key 后重新运行")
        print("""
  配置说明：
  {
    "llm": {
      "provider": "openai",          // 可选: openai / claude / deepseek
      "api_key":  "sk-xxx",          // 你的 API Key
      "model":    "gpt-4o-mini"      // 可选模型
    }
  }
""")
        return

    # 解析命令行参数
    args = sys.argv[1:]

    if '--test' in args:
        print("[测试] 检查API连通性...")
        cfg = Config(config_path)
        llm = LLMClient(cfg.llm["provider"], cfg.llm["api_key"],
                        cfg.llm.get("model"), cfg.llm.get("base_url"))
        ok = llm.test_connection()
        print(f"  API连接: {'✓ 成功' if ok else '✗ 失败'}")
        return

    try:
        pipeline = NewsIntelligencePipeline(config_path=config_path)
    except ValueError as e:
        print(f"\n  ✗ {e}")
        return

    if '--auto' in args:
        pipeline.run_full_cycle()
        return

    if '--url' in args:
        idx = args.index('--url')
        urls = [a for a in args[idx+1:] if not a.startswith('--')]
        if urls:
            pipeline.run_from_urls(urls)
        else:
            print("  请在 --url 后提供URL")
        return

    # 交互模式
    print("  已进入交互模式，输入 help 查看命令\n")
    while True:
        try:
            cmd = input("情报系统 > ").strip()
            if not cmd:
                continue
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                print("""
  命令：
    auto              自动搜索+采集+提取+汇总
    url <URL>         从指定URL提取
    text              手动输入新闻文本
    status            查看已采集数据状态
    recent            查看最近提取结果
    test              测试LLM API连通性
    quit              退出
""")
            elif cmd == 'auto':
                month = input("  目标月份 (YYYY-MM，回车=当月): ").strip()
                pipeline.run_full_cycle(target_month=month or None)

            elif cmd.startswith('url'):
                urls = cmd.split()[1:]
                if not urls:
                    urls = [input("  请输入URL: ").strip()]
                pipeline.run_from_urls(urls)

            elif cmd == 'text':
                print("  请粘贴新闻内容（输入 END 结束）:")
                lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    lines.append(line)
                if lines:
                    pipeline.run_from_text(['\n'.join(lines)])

            elif cmd == 'test':
                ok = pipeline.llm.test_connection()
                print(f"  API连接: {'✓ 成功' if ok else '✗ 失败'}")

            elif cmd == 'status':
                news_count = len(os.listdir(pipeline.store.news_dir))
                ext_count  = len(os.listdir(pipeline.store.extract_dir))
                csv_exists = os.path.exists(pipeline.store.csv_path)
                print(f"  已采集新闻: {news_count} 篇")
                print(f"  提取结果:   {ext_count} 条")
                print(f"  月度CSV:    {'✓ 存在' if csv_exists else '✗ 未生成'}")

            elif cmd == 'recent':
                recs = pipeline.store.get_recent_extractions(30)
                if not recs:
                    print("  无最近提取记录")
                else:
                    for r in recs[-5:]:
                        conf = r.get('confidence', '?')
                        src  = r.get('source_url', '?')[:50]
                        summ = r.get('market_summary', '')[:40]
                        print(f"  [{conf}] {src}...")
                        print(f"       {summ}")

        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break


if __name__ == '__main__':
    main()
