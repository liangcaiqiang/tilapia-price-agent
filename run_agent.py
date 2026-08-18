#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════
  罗非鱼鱼价预测 Agent  —— 一键启动脚本
══════════════════════════════════════════════════════════════════

整合了：
  1. tilapia_price_agent.py  —— 预测模型 Agent
  2. news_intelligence.py    —— 新闻情报采集 + LLM提取

使用方式：
  python3 run_agent.py                    # 标准启动（模拟数据）
  python3 run_agent.py --with-news        # 启动并自动采集最新新闻
  python3 run_agent.py --data data.csv    # 使用真实历史数据

首次使用新闻功能前，请先配置 config.json（填写LLM的API Key）。
"""

import os
import sys

# 将当前目录加入搜索路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tilapia_price_agent import (
    TilapiaDataGenerator, TilapiaPriceEnsemble, FeatureEngineer,
    Visualizer, TilapiaPriceAgent
)


def check_news_module():
    """检查新闻模块是否可用"""
    try:
        from news_intelligence import NewsIntelligencePipeline, Config
        config_path = os.path.join(SCRIPT_DIR, "config.json")
        if os.path.exists(config_path):
            cfg = Config(config_path)
            if cfg.llm.get("api_key"):
                return True
        return False
    except ImportError:
        return False


class IntegratedAgent(TilapiaPriceAgent):
    """
    集成新闻情报功能的增强版Agent。
    在原有命令基础上新增：
      - news_fetch    自动搜索+采集最新新闻并提取指标
      - news_url      从指定URL采集
      - news_text     手动粘贴新闻文本并提取
      - news_status   查看新闻采集状态
      - auto_update   一键 采集新闻 → 提取指标 → 更新数据 → 重训模型
    """

    HELP_EXTRA = """
╔═══════════════════════════════════════════════════════════════╗
║  📰 新闻情报命令（需要先配置 config.json 中的 API Key）        ║
╠═══════════════════════════════════════════════════════════════╣
║  news_fetch          自动搜索采集最新罗非鱼新闻并提取指标      ║
║  news_url <URL>      从指定新闻URL采集并提取                   ║
║  news_text           手动粘贴新闻内容并提取                    ║
║  news_status         查看新闻数据采集状态                      ║
║  auto_update         一键：采集→提取→更新数据→重训模型          ║
║  test_api            测试LLM API连通性                        ║
╚═══════════════════════════════════════════════════════════════╝
"""

    def __init__(self, data_path=None):
        super().__init__(data_path)

        # 尝试初始化新闻模块
        self.news_available = False
        self.pipeline = None
        try:
            from news_intelligence import NewsIntelligencePipeline, Config
            config_path = os.path.join(SCRIPT_DIR, "config.json")
            if os.path.exists(config_path):
                cfg = Config(config_path)
                if cfg.llm.get("api_key"):
                    self.pipeline = NewsIntelligencePipeline(config_path=config_path)
                    self.news_available = True
                    print("\n  ✓ 新闻情报模块已激活")
                else:
                    print("\n  ⚠ 新闻模块未激活：请在 config.json 中填写 API Key")
            else:
                print("\n  ⚠ 新闻模块未激活：缺少 config.json 配置文件")
                print("     运行 'python3 news_intelligence.py' 可生成配置模板")
        except Exception as e:
            print(f"\n  ⚠ 新闻模块加载失败: {e}")

        if self.news_available:
            print(self.HELP_EXTRA)

    def _dispatch(self, cmd):
        p = cmd.lower().split()
        a = p[0] if p else ''

        # 新闻相关命令
        if a == 'news_fetch':
            self._news_fetch()
        elif a == 'news_url':
            urls = p[1:] if len(p) > 1 else [input("  请输入新闻URL: ").strip()]
            self._news_url(urls)
        elif a == 'news_text':
            self._news_text()
        elif a == 'news_status':
            self._news_status()
        elif a == 'auto_update':
            self._auto_update()
        elif a == 'test_api':
            self._test_api()
        else:
            super()._dispatch(cmd)

    def _check_news(self):
        if not self.news_available:
            print("\n  ⚠ 新闻模块未激活。请先配置 config.json：")
            print('  {')
            print('    "llm": {')
            print('      "provider": "openai",')
            print('      "api_key":  "sk-你的key",')
            print('      "model":    "gpt-4o-mini"')
            print('    }')
            print('  }')
            return False
        return True

    def _news_fetch(self):
        if not self._check_news(): return
        month = input("  目标月份 (YYYY-MM，回车=当月): ").strip()
        self.pipeline.run_full_cycle(target_month=month or None)

    def _news_url(self, urls):
        if not self._check_news(): return
        month = input("  目标月份 (YYYY-MM，回车=当月): ").strip()
        self.pipeline.run_from_urls(urls, target_month=month or None)

    def _news_text(self):
        if not self._check_news(): return
        print("  请粘贴新闻内容（输入 END 结束）:")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        if lines:
            self.pipeline.run_from_text(['\n'.join(lines)])

    def _news_status(self):
        if not self._check_news(): return
        store = self.pipeline.store
        news_count = len([f for f in os.listdir(store.news_dir) if f.endswith('.json')])
        ext_count  = len([f for f in os.listdir(store.extract_dir) if f.endswith('.json')])
        csv_ok     = os.path.exists(store.csv_path)

        print(f"\n  📰 新闻数据状态")
        print(f"     已采集新闻:  {news_count} 篇")
        print(f"     提取结果:    {ext_count} 条")
        print(f"     月度CSV:     {'✓ ' + store.csv_path if csv_ok else '✗ 未生成'}")
        print(f"     LLM提供商:   {self.pipeline.config.llm['provider']}")
        print(f"     模型:        {self.pipeline.config.llm['model']}")

        if csv_ok:
            import pandas as pd
            df = pd.read_csv(store.csv_path)
            print(f"     CSV记录数:   {len(df)} 个月")
            if len(df) > 0:
                print(f"     时间范围:    {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    def _auto_update(self):
        """一键：采集新闻 → 提取指标 → 更新CSV → 加载到模型 → 重训"""
        if not self._check_news(): return

        print("\n" + "═"*60)
        print("  🔄 一键自动更新流程")
        print("═"*60)

        month = input("  目标月份 (YYYY-MM，回车=当月): ").strip()
        if not month:
            from datetime import datetime
            month = datetime.now().strftime("%Y-%m")

        # Step 1: 采集+提取+汇总
        monthly = self.pipeline.run_full_cycle(target_month=month)
        if not monthly:
            print("  ✗ 新闻采集/提取失败，中止更新")
            return

        # Step 2: 更新预测模型数据
        print("\n[更新模型] 将新闻提取数据注入预测模型...")
        import pandas as pd

        csv_path = self.pipeline.store.csv_path
        if os.path.exists(csv_path):
            news_df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            news_df['month'] = news_df.index.month
            news_df['year']  = news_df.index.year

            # 合并到主数据中
            for date_val, row in news_df.iterrows():
                if date_val in self.df.index:
                    # 更新已有月份
                    for col in ['us_inventory','cn_factory_inventory',
                                'pond_stock_level','fry_release_volume',
                                'feed_sales_volume','weather_risk','export_index']:
                        if col in row and pd.notna(row[col]):
                            self.df.at[date_val, col] = row[col]
                    if 'price' in row and pd.notna(row.get('price')) and row['price']:
                        try:
                            self.df.at[date_val, 'price'] = float(row['price'])
                        except:
                            pass
                else:
                    # 新增月份
                    new_row = self.df.iloc[-1].copy()
                    for col in ['us_inventory','cn_factory_inventory',
                                'pond_stock_level','fry_release_volume',
                                'feed_sales_volume','weather_risk','export_index']:
                        if col in row and pd.notna(row[col]):
                            new_row[col] = row[col]
                    if 'price' in row and pd.notna(row.get('price')) and row['price']:
                        try:
                            new_row['price'] = float(row['price'])
                        except:
                            pass
                    new_row['month'] = date_val.month
                    new_row['year']  = date_val.year
                    new_df = pd.DataFrame([new_row], index=pd.DatetimeIndex([date_val]))
                    self.df = pd.concat([self.df, new_df])

            self.df = self.df[~self.df.index.duplicated(keep='last')]
            self.df.sort_index(inplace=True)

        # Step 3: 重训模型
        print("[重训模型] ...")
        metrics = self.ensemble.fit(self.df)
        print("\n[最新验证性能]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Step 4: 显示新预测
        print("\n[更新后预测]")
        self._predict(6)

        print("\n  ✓ 一键更新完成！模型已使用最新新闻数据重新训练。")

    def _test_api(self):
        if not self._check_news(): return
        print("  正在测试API连通性...")
        ok = self.pipeline.llm.test_connection()
        print(f"  结果: {'✓ 连接成功' if ok else '✗ 连接失败'}")

    def _nlp(self, text):
        """增强的自然语言理解"""
        t = text.lower()
        if any(k in t for k in ['新闻','采集','news','抓取','爬取']):
            self._news_fetch()
        elif any(k in t for k in ['自动更新','一键','auto']):
            self._auto_update()
        elif any(k in t for k in ['测试','test','api','连接']):
            self._test_api()
        else:
            super()._nlp(text)


def main():
    args = sys.argv[1:]
    data_path = None

    # 解析参数
    if '--data' in args:
        idx = args.index('--data')
        if idx + 1 < len(args):
            data_path = args[idx + 1]

    # 启动集成Agent
    agent = IntegratedAgent(data_path=data_path)

    # 如果指定了 --with-news，自动执行一次新闻采集
    if '--with-news' in args and agent.news_available:
        print("\n[启动参数] 自动执行新闻采集...")
        agent._auto_update()

    agent.run()


if __name__ == '__main__':
    main()
