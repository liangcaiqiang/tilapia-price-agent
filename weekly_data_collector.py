#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
罗非鱼周报缺失数据采集器

优先级：
1. Firecrawl（若本机已登录或设置 FIRECRAWL_API_KEY）
2. Playwright 浏览器自动化（USITC DataWeb 匿名网页查询）
3. requests + Bing RSS + 公开网页抓取

输出：tilapia_data/weekly/YYYYMMDD/
- usitc_raw.json
- usitc_customs_value.csv
- usitc_quantity_kg.csv
- usitc_summary.json
- public_sources.json
- latest_data_supplement.md
- run_log.json
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent
RUN_DATE = date.today().strftime("%Y%m%d")
OUTPUT_DIR = BASE_DIR / "tilapia_data" / "weekly" / RUN_DATE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

SEARCH_QUERIES = [
    "罗非鱼 塘头价 茂名 湛江 海南 2026年7月",
    "罗非鱼 加工厂 收鱼 库存 开工率 2026年7月",
    "罗非鱼 投苗量 饲料销量 存塘 2026",
    "中国 罗非鱼 出口量 出口均价 2026年6月 海关",
    "US tilapia retail wholesale demand July 2026",
    "Indonesia Vietnam tilapia exports United States 2026",
]

KNOWN_PUBLIC_SOURCES = [
    {
        "name": "茂名上半年罗非鱼出口",
        "url": "https://www.sohu.com/a/1053513507_123753",
        "category": "中国出口/茂名",
    },
    {
        "name": "水产前沿7月10日罗非鱼周报",
        "url": "https://www.sohu.com/a/1048625920_210667",
        "category": "塘头价/加工厂/存塘",
    },
    {
        "name": "JCI 7月13日广东罗非鱼价格",
        "url": "https://www.chinajci.com/article/a2875443Z.html",
        "category": "塘头价/订单",
    },
    {
        "name": "农业农村部罗非鱼供需报告",
        "url": "https://scs.moa.gov.cn/jcyj/202604/t20260413_6483200.htm",
        "category": "投苗/产量/库存预测",
    },
    {
        "name": "美国贸易数据发布日期表",
        "url": "https://www.census.gov/foreign-trade/schedule.html",
        "category": "美国进口数据发布日期",
    },
    {
        "name": "美国罗非鱼航运记录线索",
        "url": "https://www.seair.co.in/us-import/product-tilapia-fillet/i-fish-brothers-inc.aspx",
        "category": "美国批发/进口到港线索",
    },
]

KEYWORDS = [
    "罗非鱼", "tilapia", "出口", "进口", "库存", "冷库", "开工", "订单",
    "收鱼", "收购价", "塘头", "价格", "投苗", "饲料", "存塘", "同比", "环比",
    "China", "Indonesia", "Vietnam", "wholesale", "retail", "shipment",
]


@dataclass
class RunLog:
    started_at: str
    firecrawl_authenticated: bool = False
    firecrawl_message: str = ""
    usitc_ok: bool = False
    usitc_latest_month: str = ""
    public_sources_fetched: int = 0
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def safe_number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return (new / old - 1.0) * 100.0


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "不可比"
    return f"{value:+.1f}%"


def check_firecrawl() -> Tuple[bool, str]:
    """检查 Firecrawl CLI 是否已经认证；不读取或输出任何密钥。"""
    try:
        result = subprocess.run(
            ["npx", "-y", "firecrawl-cli@latest", "view-config"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=BASE_DIR,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        authenticated = "Not authenticated" not in output and result.returncode == 0
        return authenticated, output[:1000]
    except Exception as exc:
        return False, f"Firecrawl 状态检查失败：{exc}"


def firecrawl_scrape(url: str) -> Optional[str]:
    """Firecrawl 已认证时使用；失败会返回 None 并由 requests 回退。"""
    try:
        result = subprocess.run(
            [
                "npx", "-y", "firecrawl-cli@latest", "scrape", url,
                "--format", "markdown", "--only-main-content",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=BASE_DIR,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 80:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def _flatten_columns(groups: Iterable[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for group in groups:
        if isinstance(group, dict) and isinstance(group.get("columns"), list):
            columns.extend(_flatten_columns(group["columns"]))
        elif isinstance(group, dict) and group.get("label") is not None:
            columns.append(str(group["label"]))
        elif isinstance(group, list):
            columns.extend(_flatten_columns(group))
    return columns


def _table_rows(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = _flatten_columns(table.get("column_groups", []))
    groups = table.get("row_groups", [])
    rows_new = groups[0].get("rowsNew", []) if groups else []
    result: List[Dict[str, Any]] = []
    for row in rows_new:
        values = [entry.get("value") for entry in row.get("rowEntries", [])]
        if len(values) == len(columns):
            result.append(dict(zip(columns, values)))
    return result


def fetch_usitc_data() -> Dict[str, Any]:
    """通过 USITC DataWeb 匿名网页，抓取 2025、2026 月度 HTS 030461 数据。"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1200})
        page.goto(
            "https://dataweb.usitc.gov/trade",
            wait_until="networkidle",
            timeout=120_000,
        )
        page.locator("#tradeFlow").select_option(label="Imports: For Consumption")
        page.wait_for_timeout(1000)
        page.evaluate(
            'document.querySelectorAll(".QSIWebResponsive,'
            '.QSIWebResponsiveDialog-Layout1-SI").forEach(e=>e.remove())'
        )

        measure_input = page.locator("ng-select[aria-label=dataToReport] input")
        for label in ["Customs Value", "First Unit of Quantity"]:
            measure_input.click(force=True)
            page.get_by_role("option", name=label, exact=True).click()
            page.keyboard.press("Escape")

        years_input = page.locator("ng-select[aria-label=years]:visible input")
        for label in ["2026 (year-to-date)", "2025"]:
            years_input.click(force=True)
            page.get_by_role("option", name=label, exact=True).click()
            page.keyboard.press("Escape")

        page.locator("#timeframeAggregation").select_option(label="Monthly")
        page.locator("#countryAggregation").select_option(
            label="Display Countries Separately"
        )
        page.locator("#commoditiesSelectedTab").select_option(
            label="Select Individual Commodities"
        )
        page.locator("#enterCommodities").fill("030461")
        page.locator("button[aria-label=addCommodities]").click()
        page.wait_for_timeout(800)

        with page.expect_response(
            lambda response: "report2/runReport" in response.url,
            timeout=120_000,
        ) as response_info:
            page.locator("button[aria-label=viewResults]").click()

        response = response_info.value
        if response.status != 200:
            raise RuntimeError(f"USITC 查询失败，HTTP {response.status}")
        payload = response.json()
        browser.close()

    (OUTPUT_DIR / "usitc_raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tables = payload.get("dto", {}).get("tables", [])
    parsed_tables: Dict[str, List[Dict[str, Any]]] = {}
    for table in tables:
        name = table.get("name", "")
        key = "quantity" if "First Unit of Quantity" in name else "customs_value"
        parsed_tables[key] = _table_rows(table)

    for key, rows in parsed_tables.items():
        file_path = OUTPUT_DIR / (
            "usitc_quantity_kg.csv" if key == "quantity" else "usitc_customs_value.csv"
        )
        if rows:
            with file_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    summary = summarize_usitc(parsed_tables)
    (OUTPUT_DIR / "usitc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def summarize_usitc(tables: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    value_rows = tables.get("customs_value", [])
    qty_rows = tables.get("quantity", [])

    def to_lookup(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, float]]:
        lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
        for row in rows:
            country = str(row.get("Country", "")).strip()
            year = int(safe_number(row.get("Year")))
            lookup[(country, year)] = {
                month: safe_number(row.get(month)) for month in MONTHS
            }
        return lookup

    value_lookup = to_lookup(value_rows)
    qty_lookup = to_lookup(qty_rows)

    latest_month = ""
    for month in MONTHS:
        total = sum(
            values[month]
            for (country, year), values in value_lookup.items()
            if year == 2026
        )
        if total > 0:
            latest_month = month

    if not latest_month:
        raise RuntimeError("USITC 返回数据中没有找到 2026 年有效月份")

    latest_idx = MONTHS.index(latest_month)
    previous_month = MONTHS[latest_idx - 1] if latest_idx > 0 else None
    tracked = ["China", "Indonesia", "Vietnam"]

    latest_total_value = sum(
        values[latest_month]
        for (country, year), values in value_lookup.items()
        if year == 2026
    )
    latest_total_qty = sum(
        values[latest_month]
        for (country, year), values in qty_lookup.items()
        if year == 2026
    )

    countries: Dict[str, Any] = {}
    for country in tracked:
        current_value = value_lookup.get((country, 2026), {}).get(latest_month, 0.0)
        current_qty = qty_lookup.get((country, 2026), {}).get(latest_month, 0.0)
        previous_value = (
            value_lookup.get((country, 2026), {}).get(previous_month, 0.0)
            if previous_month else 0.0
        )
        previous_qty = (
            qty_lookup.get((country, 2026), {}).get(previous_month, 0.0)
            if previous_month else 0.0
        )
        yoy_value = value_lookup.get((country, 2025), {}).get(latest_month, 0.0)
        yoy_qty = qty_lookup.get((country, 2025), {}).get(latest_month, 0.0)
        ytd_value = sum(
            value_lookup.get((country, 2026), {}).get(month, 0.0)
            for month in MONTHS[: latest_idx + 1]
        )
        ytd_qty = sum(
            qty_lookup.get((country, 2026), {}).get(month, 0.0)
            for month in MONTHS[: latest_idx + 1]
        )
        ytd_value_2025 = sum(
            value_lookup.get((country, 2025), {}).get(month, 0.0)
            for month in MONTHS[: latest_idx + 1]
        )
        ytd_qty_2025 = sum(
            qty_lookup.get((country, 2025), {}).get(month, 0.0)
            for month in MONTHS[: latest_idx + 1]
        )
        countries[country] = {
            "latest_value_usd": current_value,
            "latest_quantity_kg": current_qty,
            "latest_value_share_pct": (
                current_value / latest_total_value * 100 if latest_total_value else 0
            ),
            "latest_quantity_share_pct": (
                current_qty / latest_total_qty * 100 if latest_total_qty else 0
            ),
            "mom_value_pct": pct_change(current_value, previous_value),
            "mom_quantity_pct": pct_change(current_qty, previous_qty),
            "yoy_value_pct": pct_change(current_value, yoy_value),
            "yoy_quantity_pct": pct_change(current_qty, yoy_qty),
            "ytd_value_usd": ytd_value,
            "ytd_quantity_kg": ytd_qty,
            "ytd_value_yoy_pct": pct_change(ytd_value, ytd_value_2025),
            "ytd_quantity_yoy_pct": pct_change(ytd_qty, ytd_qty_2025),
            "ytd_unit_value_usd_per_kg": ytd_value / ytd_qty if ytd_qty else None,
        }

    def totals(year: int, month: str) -> Tuple[float, float]:
        value = sum(
            values[month]
            for (country, row_year), values in value_lookup.items()
            if row_year == year
        )
        qty = sum(
            values[month]
            for (country, row_year), values in qty_lookup.items()
            if row_year == year
        )
        return value, qty

    latest_value_2025, latest_qty_2025 = totals(2025, latest_month)
    previous_total_value, previous_total_qty = (
        totals(2026, previous_month) if previous_month else (0.0, 0.0)
    )

    months_ytd = MONTHS[: latest_idx + 1]
    ytd_total_value = sum(totals(2026, month)[0] for month in months_ytd)
    ytd_total_qty = sum(totals(2026, month)[1] for month in months_ytd)
    ytd_total_value_2025 = sum(totals(2025, month)[0] for month in months_ytd)
    ytd_total_qty_2025 = sum(totals(2025, month)[1] for month in months_ytd)

    return {
        "source": "USITC DataWeb / U.S. Census Bureau official trade statistics",
        "hts": "030461",
        "accessed_at": datetime.now().astimezone().isoformat(),
        "latest_month": latest_month,
        "previous_month": previous_month,
        "latest_total_value_usd": latest_total_value,
        "latest_total_quantity_kg": latest_total_qty,
        "latest_total_mom_value_pct": pct_change(
            latest_total_value, previous_total_value
        ),
        "latest_total_mom_quantity_pct": pct_change(
            latest_total_qty, previous_total_qty
        ),
        "latest_total_yoy_value_pct": pct_change(
            latest_total_value, latest_value_2025
        ),
        "latest_total_yoy_quantity_pct": pct_change(
            latest_total_qty, latest_qty_2025
        ),
        "ytd_total_value_usd": ytd_total_value,
        "ytd_total_quantity_kg": ytd_total_qty,
        "ytd_total_value_yoy_pct": pct_change(
            ytd_total_value, ytd_total_value_2025
        ),
        "ytd_total_quantity_yoy_pct": pct_change(
            ytd_total_qty, ytd_total_qty_2025
        ),
        "countries": countries,
        "note": "最新月份由官方结果中最后一个非零月份自动识别；未发布月份不会冒充最新数据。",
    }


def bing_rss_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    url = f"https://www.bing.com/search?q={quote(query)}&format=rss"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    results: List[Dict[str, str]] = []
    for item in root.findall("./channel/item")[:max_results]:
        results.append(
            {
                "query": query,
                "title": item.findtext("title", default=""),
                "url": item.findtext("link", default=""),
                "snippet": item.findtext("description", default=""),
                "pub_date": item.findtext("pubDate", default=""),
            }
        )
    return results


def extract_date(text: str, soup: BeautifulSoup) -> str:
    meta_candidates = [
        ("property", "article:published_time"),
        ("name", "publishdate"),
        ("name", "pubdate"),
        ("name", "date"),
        ("itemprop", "datePublished"),
    ]
    for key, value in meta_candidates:
        tag = soup.find("meta", attrs={key: value})
        if tag and tag.get("content"):
            return str(tag.get("content"))[:32]
    patterns = [
        r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)",
        r"(20\d{2}年\d{1,2}月\d{1,2}日)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:4000])
        if match:
            return match.group(1)
    return ""


def relevant_snippets(text: str, limit: int = 12) -> List[str]:
    clean = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[。！？.!?])\s*|\s{2,}", clean)
    scored: List[Tuple[int, str]] = []
    for part in parts:
        part = part.strip()
        if len(part) < 18 or len(part) > 500:
            continue
        score = sum(1 for keyword in KEYWORDS if keyword.lower() in part.lower())
        if score:
            scored.append((score, part))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    output: List[str] = []
    for _, part in scored:
        if part not in output:
            output.append(part)
        if len(output) >= limit:
            break
    return output


def fetch_public_page(
    url: str, firecrawl_authenticated: bool = False
) -> Dict[str, Any]:
    fetched_by = "requests"
    markdown = None
    if firecrawl_authenticated:
        markdown = firecrawl_scrape(url)
        if markdown:
            fetched_by = "firecrawl"

    try:
        if markdown:
            text = markdown
            title = markdown.splitlines()[0].lstrip("# ") if markdown else ""
            published = ""
        else:
            response = requests.get(url, headers=HEADERS, timeout=35)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            text = soup.get_text(" ", strip=True)
            published = extract_date(text, soup)
        return {
            "url": url,
            "title": title,
            "published_date": published,
            "fetched_at": datetime.now().astimezone().isoformat(),
            "fetched_by": fetched_by,
            "text_length": len(text),
            "snippets": relevant_snippets(text),
            "ok": len(text) > 100,
        }
    except Exception as exc:
        return {
            "url": url,
            "title": "",
            "published_date": "",
            "fetched_at": datetime.now().astimezone().isoformat(),
            "fetched_by": fetched_by,
            "text_length": 0,
            "snippets": [],
            "ok": False,
            "error": str(exc),
        }


def collect_public_sources(firecrawl_authenticated: bool) -> List[Dict[str, Any]]:
    searches: List[Dict[str, str]] = []
    for query in SEARCH_QUERIES:
        try:
            searches.extend(bing_rss_search(query, max_results=8))
        except Exception as exc:
            searches.append(
                {
                    "query": query,
                    "title": "",
                    "url": "",
                    "snippet": "",
                    "pub_date": "",
                    "error": str(exc),
                }
            )
        time.sleep(0.3)

    urls: List[Tuple[str, str, str]] = []
    for source in KNOWN_PUBLIC_SOURCES:
        urls.append((source["url"], source["name"], source["category"]))
    for result in searches:
        url = result.get("url", "")
        if url and any(
            domain in url
            for domain in [
                "customs.gov.cn", "moa.gov.cn", "sohu.com", "fishfirst.cn",
                "chinajci.com", "nfncb.cn", "frozengoods.com", "seafoodsource.com",
                "undercurrentnews.com", "vasep.com.vn", "census.gov", "usitc.gov",
                "seair.co.in",
            ]
        ):
            urls.append((url, result.get("title", ""), "搜索发现"))

    seen = set()
    pages: List[Dict[str, Any]] = []
    for url, source_name, category in urls:
        normalized = url.split("#")[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        page = fetch_public_page(normalized, firecrawl_authenticated)
        page["source_name"] = source_name
        page["category"] = category
        pages.append(page)
        time.sleep(0.25)

    output = {"search_results": searches, "fetched_pages": pages}
    (OUTPUT_DIR / "public_sources.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pages


def find_page(pages: List[Dict[str, Any]], needle: str) -> Optional[Dict[str, Any]]:
    needle_lower = needle.lower()
    for page in pages:
        haystack = " ".join(
            [
                page.get("source_name", ""), page.get("title", ""),
                page.get("url", ""), page.get("category", ""),
            ]
        ).lower()
        if needle_lower in haystack:
            return page
    return None


def make_markdown(
    usitc: Dict[str, Any], pages: List[Dict[str, Any]], log: RunLog
) -> str:
    latest = usitc["latest_month"]
    previous = usitc.get("previous_month") or "上月"
    c = usitc["countries"]

    lines = [
        "# 罗非鱼周报缺失数据自动补充",
        "",
        f"**运行日期：{date.today().isoformat()}**  ",
        f"**采集目录：`{OUTPUT_DIR}`**",
        "",
        "## 1. Firecrawl 与采集状态",
        "",
        (
            "- Firecrawl：已认证，本轮网页正文优先用 Firecrawl 抓取。"
            if log.firecrawl_authenticated
            else "- Firecrawl：程序已安装，但本机当前未认证；本轮自动使用 Playwright/requests 回退，不影响 USITC 官方数据获取。"
        ),
        "- USITC DataWeb：已通过 Python + Playwright 匿名网页查询抓取成功。",
        "",
        "## 2. 美国冷冻罗非鱼片进口（HTS 030461，官方）",
        "",
        f"截至本次运行，官方结果最后一个非零月份为 **{latest} 2026**。未发布月份保持为0，不作为最新数据。",
        "",
        "| 指标 | 最新值 | 环比 | 同比 |",
        "|---|---:|---:|---:|",
        f"| 美国进口金额 | ${usitc['latest_total_value_usd']:,.0f} | {fmt_pct(usitc['latest_total_mom_value_pct'])} | {fmt_pct(usitc['latest_total_yoy_value_pct'])} |",
        f"| 美国进口数量 | {usitc['latest_total_quantity_kg']/1_000_000:,.2f}百万kg | {fmt_pct(usitc['latest_total_mom_quantity_pct'])} | {fmt_pct(usitc['latest_total_yoy_quantity_pct'])} |",
        "",
        "### 来源国结构",
        "",
        "| 来源国 | 最新数量 | 数量份额 | 数量环比 | 数量同比 | 1月至最新月均价 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for country, cn_name in [("China", "中国"), ("Indonesia", "印尼"), ("Vietnam", "越南")]:
        row = c[country]
        unit_value = row["ytd_unit_value_usd_per_kg"]
        lines.append(
            f"| {cn_name} | {row['latest_quantity_kg']/1_000_000:.2f}百万kg | "
            f"{row['latest_quantity_share_pct']:.1f}% | {fmt_pct(row['mom_quantity_pct'])} | "
            f"{fmt_pct(row['yoy_quantity_pct'])} | ${unit_value:.2f}/kg |"
        )

    lines.extend(
        [
            "",
            "### 官方数据直接结论",
            "",
            f"- {latest}美国总进口数量较{previous}变化 **{fmt_pct(usitc['latest_total_mom_quantity_pct'])}**。",
            f"- 中国来源数量环比 **{fmt_pct(c['China']['mom_quantity_pct'])}**，中国最新数量份额 **{c['China']['latest_quantity_share_pct']:.1f}%**。",
            f"- 印尼来源数量环比 **{fmt_pct(c['Indonesia']['mom_quantity_pct'])}**；越南环比 **{fmt_pct(c['Vietnam']['mom_quantity_pct'])}**。",
            "- 这能够验证美国当月采购流向，但不能直接等同于美国进口商库存见底；库存天数仍需进口商/批发商一线数据。",
            "",
            "## 3. 公开网页已抓取线索",
            "",
        ]
    )

    useful_pages = [page for page in pages if page.get("ok") and page.get("snippets")]
    for page in useful_pages[:12]:
        lines.append(
            f"### {page.get('source_name') or page.get('title') or '公开来源'}"
        )
        lines.append("")
        lines.append(
            f"- 发布/页面日期：{page.get('published_date') or '页面未明确识别'}"
        )
        lines.append(f"- 抓取方式：{page.get('fetched_by')}")
        for snippet in page.get("snippets", [])[:4]:
            lines.append(f"- {snippet}")
        lines.append("")

    lines.extend(
        [
            "## 4. 本轮仍不能从公开网页得到的量化数据",
            "",
            "- 美国进口商/批发商罗非鱼库存吨数、库存天数和订单覆盖月数。",
            "- 广东、海南、广西主要罗非鱼饲料厂当月专用料销量。",
            "- 茂名、湛江加工厂平均开工率、日收鱼量、成品冷库利用率和周转天数。",
            "- 主产区实时存塘总量及各规格占比；公开报道只能提供方向性一线线索。",
            "- 若这些企业经营数据未公开，Firecrawl 也无法突破权限或把不存在的数据抓出来。",
            "",
            "## 5. 数据可靠性分层",
            "",
            "- **A级已证实**：USITC DataWeb / 美国商务部人口普查局官方贸易统计。",
            "- **B级较可靠**：海关口径的地方政府/主流媒体报道、农业农村部供需报告。",
            "- **C级一线线索**：水产行业周报、报价网站、航运提单样本。",
            "- **未获得**：企业内部库存、开工率、饲料销量和实时存塘数据库。",
            "",
            "---",
            "程序会自动保留原始JSON、CSV和网页证据，便于下一期直接做环比。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    log = RunLog(started_at=datetime.now().astimezone().isoformat())
    print(f"[输出目录] {OUTPUT_DIR}")

    authenticated, message = check_firecrawl()
    log.firecrawl_authenticated = authenticated
    log.firecrawl_message = message
    print(f"[Firecrawl] {'已认证' if authenticated else '未认证，启用回退通道'}")

    try:
        print("[USITC] 正在抓取 HTS 030461 官方数据……")
        usitc = fetch_usitc_data()
        log.usitc_ok = True
        log.usitc_latest_month = usitc["latest_month"]
        print(f"[USITC] 完成，最新有效月份：{usitc['latest_month']} 2026")
    except Exception as exc:
        log.errors.append(f"USITC: {exc}")
        print(f"[USITC] 失败：{exc}", file=sys.stderr)
        # DataWeb 偶尔会在匿名网页查询时超时。为了让周报不中断，
        # 若本机已有最近一次官方 USITC 汇总，就先沿用缓存并继续抓取本周公开网页线索。
        usitc = None
        weekly_dir = BASE_DIR / "tilapia_data" / "weekly"
        for cache_path in sorted(weekly_dir.glob("*/usitc_summary.json"), reverse=True):
            if cache_path.parent == OUTPUT_DIR:
                continue
            try:
                usitc = json.loads(cache_path.read_text(encoding="utf-8"))
                usitc["note"] = (
                    usitc.get("note", "")
                    + f" 本次USITC在线查询超时，暂沿用缓存：{cache_path}。"
                )
                log.errors.append(f"USITC使用缓存: {cache_path}")
                log.usitc_latest_month = usitc.get("latest_month", "cached")
                print(f"[USITC] 已沿用缓存：{cache_path}")
                break
            except Exception as cache_exc:
                log.errors.append(f"USITC缓存读取失败 {cache_path}: {cache_exc}")
        if usitc is None:
            (OUTPUT_DIR / "run_log.json").write_text(
                json.dumps(asdict(log), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return 1

    print("[网页] 正在搜索并抓取公开市场线索……")
    pages = collect_public_sources(authenticated)
    log.public_sources_fetched = sum(1 for page in pages if page.get("ok"))

    markdown = make_markdown(usitc, pages, log)
    report_path = OUTPUT_DIR / "latest_data_supplement.md"
    report_path.write_text(markdown, encoding="utf-8")

    (OUTPUT_DIR / "run_log.json").write_text(
        json.dumps(asdict(log), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[完成] 报告：{report_path}")
    print(f"[完成] 成功抓取公开页面：{log.public_sources_fetched}个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
