#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Firecrawl Python SDK 定向补抓 20260803 周报缺失数据。"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from firecrawl import Firecrawl

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "tilapia_data" / "weekly" / "20260803"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    # 第4部分报告明确缺失的数据
    "2026年8月 罗非鱼 加工厂 库存 开工率 日收鱼量 茂名 湛江",
    "2026年7月 罗非鱼 加工厂 库存 开工率 日收鱼量 茂名 湛江",
    "2026年8月 罗非鱼 饲料 销量 罗非鱼料 广东 海南 广西",
    "2026年7月 罗非鱼 饲料 销量 罗非鱼料 广东 海南 广西",
    "2026年8月 罗非鱼 存塘 规格结构 广东 海南 茂名 湛江",
    "2026年7月 罗非鱼 存塘 规格结构 广东 海南 茂名 湛江",
    "2026年8月 罗非鱼 塘头价 加工厂收购价 茂名 湛江 海南",
    "2026年7月 罗非鱼 塘头价 加工厂收购价 茂名 湛江 海南",
    "US tilapia importer inventory cold storage wholesale demand August 2026",
    "US frozen tilapia fillet wholesale price August 2026",
]

KNOWN_URLS = [
    "https://www.tensfish.com/zixunredian/811.html",
    "https://www.sohu.com/a/1048625920_210667",
    "https://www.sohu.com/a/1053513507_123753",
    "https://www.chinajci.com/article/a2875443Z.html",
    "https://scs.moa.gov.cn/jcyj/202604/t20260413_6483200.htm",
    "https://www.foodomarket.com/en-us/products/frozen-fish-and-seafood/frozen-tilapia-fillet",
]

KEYWORDS = [
    "库存", "冷库", "开工", "日收", "收鱼", "加工厂", "订单", "塘头", "收购价",
    "投苗", "苗", "饲料", "罗非鱼料", "存塘", "规格", "缺鱼", "供应", "走货",
    "inventory", "cold storage", "wholesale", "retail", "demand", "tilapia",
]

MISSING_ITEMS = [
    "美国进口商/批发商罗非鱼库存吨数、库存天数和订单覆盖月数",
    "广东、海南、广西主要罗非鱼饲料厂当月专用料销量",
    "茂名、湛江加工厂平均开工率、日收鱼量、成品冷库利用率和周转天数",
    "主产区实时存塘总量及各规格占比",
]


def load_key() -> str:
    env_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if env_key:
        return env_key
    return subprocess.check_output(
        ["security", "find-generic-password", "-a", "caiqiang2019", "-s", "firecrawl-api-key", "-w"],
        text=True,
    ).strip()


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def relevant_lines(markdown: str, limit: int = 12) -> list[str]:
    text = compact_text(markdown)
    parts = re.split(r"(?<=[。！？.!?])\s*", text)
    scored: list[tuple[int, str]] = []
    for part in parts:
        if len(part) < 16 or len(part) > 420:
            continue
        score = sum(1 for k in KEYWORDS if k.lower() in part.lower())
        if score:
            scored.append((score, part))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    out: list[str] = []
    for _, line in scored:
        if line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def get_doc_text(doc: dict[str, Any]) -> str:
    return doc.get("markdown") or doc.get("description") or doc.get("title") or ""


def run_searches(fc: Firecrawl) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, query in enumerate(QUERIES, 1):
        print(f"[Firecrawl搜索 {i}/{len(QUERIES)}] {query}")
        try:
            res = fc.search(
                query,
                sources=["web"],
                limit=2,
                tbs="qdr:m",
                scrape_options={"formats": ["markdown"], "only_main_content": True},
                timeout=60_000,
            )
            raw = to_jsonable(res)
            results.append({"query": query, "ok": True, "result": raw})
        except Exception as exc:
            results.append({"query": query, "ok": False, "error": str(exc)})
    return results


def run_scrapes(fc: Firecrawl) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for i, url in enumerate(KNOWN_URLS, 1):
        print(f"[Firecrawl抓取 {i}/{len(KNOWN_URLS)}] {url}")
        try:
            doc = fc.scrape(url, formats=["markdown"], only_main_content=True, timeout=60_000)
            raw = to_jsonable(doc)
            text = get_doc_text(raw)
            pages.append({
                "url": url,
                "ok": True,
                "title": (raw.get("metadata") or {}).get("title") or raw.get("title") or "",
                "text_length": len(text),
                "snippets": relevant_lines(text),
                "raw": raw,
            })
        except Exception as exc:
            pages.append({"url": url, "ok": False, "error": str(exc)})
    return pages


def flatten_search_hits(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen = set()
    for item in searches:
        if not item.get("ok"):
            continue
        result = item.get("result") or {}
        for bucket in ["web", "news"]:
            for hit in result.get(bucket) or []:
                url = hit.get("url") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                text = get_doc_text(hit)
                hits.append({
                    "query": item.get("query"),
                    "url": url,
                    "title": hit.get("title") or "",
                    "description": hit.get("description") or "",
                    "snippets": relevant_lines(text),
                    "text_length": len(text),
                })
    return hits


def classify_findings(hits: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    all_items = hits + pages
    categories = {
        "加工厂库存/开工/收鱼": [],
        "饲料销量": [],
        "存塘/规格结构": [],
        "塘头价/加工厂收购价": [],
        "美国库存/批发需求": [],
    }
    patterns = {
        "加工厂库存/开工/收鱼": ["库存", "开工", "日收", "收鱼", "无米下锅", "做两天", "停三天", "加工厂"],
        "饲料销量": ["饲料销量", "罗非鱼料", "专用料", "投料"],
        "存塘/规格结构": ["存塘", "规格", "缺鱼", "投苗", "苗量", "商品鱼"],
        "塘头价/加工厂收购价": ["塘头", "收购价", "元/斤", "报价"],
        "美国库存/批发需求": ["inventory", "cold storage", "wholesale", "retail", "demand", "price", "tilapia"],
    }
    for item in all_items:
        blob = " ".join([item.get("title", ""), item.get("description", ""), " ".join(item.get("snippets", []))])
        low = blob.lower()
        for cat, keys in patterns.items():
            if any(k.lower() in low for k in keys):
                if item not in categories[cat]:
                    categories[cat].append(item)
    return categories


def write_report(searches: list[dict[str, Any]], pages: list[dict[str, Any]], credits_before: dict[str, Any], credits_after: dict[str, Any]) -> None:
    hits = flatten_search_hits(searches)
    categories = classify_findings(hits, pages)
    lines: list[str] = []
    lines.append("# Firecrawl SDK 补抓缺失数据结果（2026-08-03）")
    lines.append("")
    lines.append(f"**运行时间：{datetime.now().astimezone().isoformat()}**  ")
    lines.append(f"**输出目录：`{OUT_DIR}`**  ")
    lines.append(f"**Firecrawl额度：运行前 {credits_before.get('remaining_credits')}，运行后 {credits_after.get('remaining_credits')}**")
    lines.append("")
    lines.append("## 1. 本次专门补抓的缺口")
    for x in MISSING_ITEMS:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 2. 补抓结论")
    lines.append("")
    lines.append("### 2.1 加工厂库存、开工、日收鱼量")
    if categories["加工厂库存/开工/收鱼"]:
        lines.append("- **补到一线线索，但没有补到官方量化吨数/开工率。**")
    else:
        lines.append("- **未补到新的公开有效线索。**")
    lines.append("- 已有公开线索仍集中在：加工端库存偏紧、开工不足、部分工厂做两天停三天、供应下降、提价收鱼。")
    lines.append("- 仍缺：日收鱼量、平均开工率、冷库利用率、库存周转天数。")
    lines.append("")
    lines.append("### 2.2 饲料销量")
    if categories["饲料销量"]:
        lines.append("- Firecrawl 找到与投料/饲料相关的网页线索，但**没有找到海大、通威、恒兴、粤海等企业的罗非鱼专用料月度销量公开数据**。")
    else:
        lines.append("- **未找到罗非鱼专用料月度销量公开数据。**")
    lines.append("- 这部分应继续靠内部销量表、经销商周报或饲料厂一线数据补充。")
    lines.append("")
    lines.append("### 2.3 存塘量与规格结构")
    if categories["存塘/规格结构"]:
        lines.append("- **补到方向性线索**：缺鱼、投苗下降、存塘偏紧等，但没有找到主产区实时存塘总量和规格结构表。")
    else:
        lines.append("- **未找到实时存塘总量和规格结构公开表。**")
    lines.append("- 可用替代指标：投苗同比、塘头价、加工厂收鱼难度、活鱼走货量。")
    lines.append("")
    lines.append("### 2.4 美国库存/批发需求")
    if categories["美国库存/批发需求"]:
        lines.append("- **补到美国批发/零售价格或需求线索**，但没有找到进口商库存吨数、库存天数、订单覆盖月数。")
    else:
        lines.append("- **未找到美国进口商库存吨数、库存天数、订单覆盖月数。**")
    lines.append("- 当前仍需以 USITC 进口数量、批发报价和提单样本作为替代指标。")
    lines.append("")
    lines.append("## 3. 分类证据摘录")
    for cat, items in categories.items():
        lines.append("")
        lines.append(f"### {cat}")
        if not items:
            lines.append("- 未找到可用摘录。")
            continue
        for item in items[:6]:
            url = item.get("url", "")
            title = item.get("title") or item.get("source_name") or "未识别标题"
            lines.append(f"- **{title}**  ")
            lines.append(f"  URL：{url}")
            for s in item.get("snippets", [])[:4]:
                lines.append(f"  - {s}")
    lines.append("")
    lines.append("## 4. 数据等级判断")
    lines.append("- A级：USITC、海关、农业农村部等官方统计。")
    lines.append("- B级：地方政府/主流媒体转述海关或官方部门数据。")
    lines.append("- C级：行业媒体、一线采访、报价网站、提单样本。")
    lines.append("- D级：泛资讯、旧数据、与本周价格判断弱相关的信息。")
    lines.append("")
    lines.append("## 5. 下一步建议")
    lines.append("- 将 `firecrawl_missing_data_supplement.md` 作为今天周报的补充附件。")
    lines.append("- 若要把缺失数据真正量化，需要接入内部销量表、加工厂收鱼表、经销商报价表或人工填写模板。")
    lines.append("- 下次可把 Firecrawl SDK 集成进 `weekly_data_collector.py`，避免再显示“未认证”。")
    (OUT_DIR / "firecrawl_missing_data_supplement.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    key = load_key()
    fc = Firecrawl(api_key=key, timeout=120)
    credits_before = to_jsonable(fc.get_credit_usage())
    print(f"[Firecrawl] SDK认证成功，剩余额度：{credits_before.get('remaining_credits')}")
    searches = run_searches(fc)
    pages = run_scrapes(fc)
    credits_after = to_jsonable(fc.get_credit_usage())
    raw = {
        "started_at": datetime.now().astimezone().isoformat(),
        "credits_before": credits_before,
        "credits_after": credits_after,
        "queries": QUERIES,
        "known_urls": KNOWN_URLS,
        "searches": searches,
        "scraped_pages": pages,
    }
    (OUT_DIR / "firecrawl_missing_data_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(searches, pages, credits_before, credits_after)
    print(f"[完成] {OUT_DIR / 'firecrawl_missing_data_supplement.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
