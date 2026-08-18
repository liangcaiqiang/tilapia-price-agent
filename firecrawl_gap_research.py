#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Firecrawl Python SDK 补齐罗非鱼周报公开数据缺口。"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

from firecrawl import Firecrawl

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "tilapia_data" / "firecrawl_gap" / date.today().strftime("%Y%m%d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "2026年6月 中国 罗非鱼 出口量 出口额 出口均价 海关",
    "2026 China tilapia exports June 2026 volume value average price customs",
    "2026 Indonesia tilapia exports United States monthly imports",
    "2026 Vietnam tilapia exports United States monthly imports",
    "US tilapia retail sales wholesale demand June July 2026",
    "US frozen tilapia fillet wholesale price July 2026",
    "2026年7月 罗非鱼 塘头价 茂名 湛江 海南 加工厂收购价",
    "2026年7月 茂名 罗非鱼 加工厂 库存 开工 收鱼",
    "2026 罗非鱼 投苗量 同比 广东 海南 广西",
    "2026 罗非鱼 饲料销量 广东 海南 广西",
    "2026年7月 罗非鱼 存塘 规格结构 广东 海南 茂名",
    "2026 罗非鱼 冷库库存 加工厂成品库存 茂名 湛江",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "research_as_of": {"type": "string"},
        "data_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string"},
                    "value_text": {"type": ["string", "null"]},
                    "period": {"type": ["string", "null"]},
                    "comparison": {"type": ["string", "null"]},
                    "publication_date": {"type": ["string", "null"]},
                    "source_name": {"type": ["string", "null"]},
                    "source_url": {"type": ["string", "null"]},
                    "evidence": {"type": ["string", "null"]},
                    "reliability": {
                        "type": "string",
                        "enum": ["official", "authoritative_secondary", "industry_line", "weak_secondary"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["confirmed", "line_clue", "not_updated", "unavailable"],
                    },
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["indicator", "reliability", "status"],
            },
        },
        "unavailable_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string"},
                    "reason": {"type": "string"},
                    "best_available_proxy": {"type": ["string", "null"]},
                },
                "required": ["indicator", "reason"],
            },
        },
        "key_findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["research_as_of", "data_points", "unavailable_items", "key_findings"],
}

PROMPT = """
你是罗非鱼国际贸易和华南养殖市场研究员。请检索截至2026年7月27日可公开获得的最新数据，专门补齐以下周报缺口：
1. 中国罗非鱼2026年6月或最新月出口量、出口额、出口均价；
2. 印尼和越南对美国出口/美国自两国进口的最新月度数量、金额和份额；
3. 美国罗非鱼零售销量、批发需求、批发价格、进口商库存或补库线索；
4. 广东、海南、茂名、湛江最新塘头价和加工厂收购价、收鱼量、开工率；
5. 罗非鱼投苗量、专用饲料销量、主要产区存塘量及规格结构；
6. 茂名、湛江加工厂成品库存、冷库库存或库存周转线索。

严格要求：
- 只记录网页明确给出的事实，不得用常识补数或推断数字。
- 每个数据点必须给出数据期、环比/同比（原文有才写）、来源名称、URL、发布日期和原文证据。
- 官方统计优先；地方政府/主流媒体转述海关列为权威二手；行业媒体和从业者采访列为一线线索。
- 必须区分 confirmed、line_clue、not_updated、unavailable。
- 旧数据只能标注为“最新公开但非本周”，不能冒充本周数据。
- 对企业内部不公开的库存、开工率、饲料销量，找不到就明确 unavailable，不得编造。
- 搜索中文、英文、越南语和印尼语来源。
"""


def load_key() -> str:
    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if key:
        return key
    return subprocess.check_output(
        [
            "security", "find-generic-password", "-a", "caiqiang2019",
            "-s", "firecrawl-api-key", "-w",
        ],
        text=True,
    ).strip()


def jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [jsonable(value) for value in obj]
    return obj


def run_searches(client: Firecrawl) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, query in enumerate(QUERIES, 1):
        print(f"[搜索 {index}/{len(QUERIES)}] {query}")
        try:
            response = client.search(
                query,
                sources=["web", "news"],
                limit=3,
                tbs="qdr:m",
                scrape_options={
                    "formats": ["markdown"],
                    "only_main_content": True,
                    "max_age": 24 * 60 * 60 * 1000,
                },
                timeout=90_000,
            )
            results.append({"query": query, "ok": True, "result": jsonable(response)})
        except Exception as exc:
            results.append({"query": query, "ok": False, "error": str(exc)})
    return results


def write_markdown(agent_result: dict[str, Any], searches: list[dict[str, Any]], credits: dict[str, Any]) -> None:
    data = agent_result.get("data") or {}
    lines = [
        "# Firecrawl 罗非鱼周报缺口研究结果",
        "",
        f"**运行时间：{datetime.now().astimezone().isoformat()}**  ",
        f"**运行前剩余额度：{credits.get('remaining_credits', '未知')}**",
        "",
        "## 一、结构化数据点",
        "",
        "| 指标 | 数值/状态 | 数据期 | 变化 | 来源与发布日期 | 证据等级 |",
        "|---|---|---|---|---|---|",
    ]
    for item in data.get("data_points", []):
        source = item.get("source_name") or "未识别"
        if item.get("publication_date"):
            source += f"（{item['publication_date']}）"
        if item.get("source_url"):
            source += f" {item['source_url']}"
        value = item.get("value_text") or item.get("status") or "无"
        lines.append(
            "| {indicator} | {value} | {period} | {comparison} | {source} | {reliability}/{status} |".format(
                indicator=str(item.get("indicator", "")).replace("|", "／"),
                value=str(value).replace("|", "／"),
                period=str(item.get("period") or "未注明").replace("|", "／"),
                comparison=str(item.get("comparison") or "未注明").replace("|", "／"),
                source=str(source).replace("|", "／"),
                reliability=item.get("reliability", ""),
                status=item.get("status", ""),
            )
        )
        if item.get("evidence"):
            lines.append(f"\n> {item['indicator']}证据：{item['evidence']}\n")
        if item.get("notes"):
            lines.append(f"> 说明：{item['notes']}\n")

    lines.extend(["", "## 二、仍无法公开获得的数据", ""])
    for item in data.get("unavailable_items", []):
        lines.append(f"- **{item.get('indicator')}**：{item.get('reason')}")
        if item.get("best_available_proxy"):
            lines.append(f"  - 可用替代指标：{item['best_available_proxy']}")

    lines.extend(["", "## 三、关键发现", ""])
    for finding in data.get("key_findings", []):
        lines.append(f"- {finding}")

    lines.extend([
        "",
        "## 四、Firecrawl 原始搜索结果",
        "",
        f"共执行 {len(searches)} 个定向查询；完整正文与元数据见 `search_results.json`。",
        "",
        "## 五、使用注意",
        "",
        "- Firecrawl 可以抓取公开网页，但不能突破订阅墙、登录权限或企业内部数据库。",
        "- 一线媒体关于投苗、存塘、库存的比例应继续通过苗场、饲料厂和加工厂交叉验证。",
        "- 官方月度数据未发布时，程序会标注 not_updated，不使用旧数据冒充本周数据。",
    ])
    (OUT_DIR / "firecrawl_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "search", "all"], default="all")
    args = parser.parse_args()

    key = load_key()
    client = Firecrawl(api_key=key, timeout=120)
    credits_before = jsonable(client.get_credit_usage())
    print(f"[Firecrawl] 连通成功，剩余额度：{credits_before.get('remaining_credits')}")

    agent_path = OUT_DIR / "agent_result.json"
    search_path = OUT_DIR / "search_results.json"
    agent_result: dict[str, Any] = {}
    searches: list[dict[str, Any]] = []

    if args.mode in {"agent", "all"}:
        print("[Agent] 正在进行多语种、证据约束的缺口研究……")
        agent_response = client.agent(
            prompt=PROMPT,
            schema=SCHEMA,
            max_credits=120,
            model="spark-1-pro",
            timeout=160,
        )
        agent_result = jsonable(agent_response)
        agent_path.write_text(
            json.dumps(agent_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif agent_path.exists():
        agent_result = json.loads(agent_path.read_text(encoding="utf-8"))

    if args.mode in {"search", "all"}:
        searches = run_searches(client)
        search_path.write_text(
            json.dumps(searches, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif search_path.exists():
        searches = json.loads(search_path.read_text(encoding="utf-8"))

    credits_after = jsonable(client.get_credit_usage())
    run_log = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "output_dir": str(OUT_DIR),
        "credits_before": credits_before,
        "credits_after": credits_after,
        "agent_success": bool(agent_result.get("success")),
        "searches_ok": sum(1 for result in searches if result.get("ok")),
        "searches_total": len(searches),
    }
    (OUT_DIR / "run_log.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if agent_result and searches:
        write_markdown(agent_result, searches, credits_before)

    print(f"[完成] 输出目录：{OUT_DIR}")
    print(f"[完成] Agent成功：{run_log['agent_success']}")
    print(f"[完成] 搜索成功：{run_log['searches_ok']}/{run_log['searches_total']}")
    print(f"[额度] 剩余：{credits_after.get('remaining_credits')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
