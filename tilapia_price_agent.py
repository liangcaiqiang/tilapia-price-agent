#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════
  罗非鱼鱼价预测 Agent  v2.0
  Tilapia Price Prediction Agent
══════════════════════════════════════════════════════════════════

【核心预测因子（行业经验 + 学术研究）】

  ★ 先行指标（Leading Indicators）— 对未来价格有预测作用
  ┌─────────────────────────────────────────────────────────┐
  │ 1. 美国收购商仓库库存    → 库存高 → 采购减少 → 价格承压   │
  │ 2. 中国加工厂仓库库存    → 库存高 → 收鱼减少 → 价格下跌   │
  │ 3. 鱼塘存塘量           → 存塘多 → 供应过剩 → 价格下跌   │
  │ 4. 苗种放苗量（T-4月）   → 多放苗 → 4月后供应↑ → 价格↓  │
  │ 5. 主流饲料厂饲料销量    → 销量高 → 养殖规模大 → 4月后↓  │
  └─────────────────────────────────────────────────────────┘

  ☆ 同期/外生因子（Concurrent Factors）
  ┌─────────────────────────────────────────────────────────┐
  │ 6. 饲料成本（豆粕/玉米）  → 成本上升 → 出塘价支撑        │
  │ 7. 季节性规律            → 7-9月集中出塘，价格最低       │
  │ 8. 极端天气（寒潮/台风）  → 死亡率↑ → 供应减少 → 价格↑  │
  │ 9. 出口指数（美国占比）   → 贸易政策影响               │
  └─────────────────────────────────────────────────────────┘

【模型架构】
  - 指数平滑 ETS          捕捉趋势与短期惯性
  - 季节性线性模型         捕捉周期性规律
  - 多特征岭回归           整合全部 9 类因子（含滞后/超前特征）
  - 加权集成              验证集MAE自动调权

【使用方式】
  python3 tilapia_price_agent.py [data.csv]

  若有真实数据：CSV列名参考 DataSchema 注释。
  无真实数据时：使用高保真模拟数据演示。

依赖：pandas, numpy, matplotlib（标准/常见库，无需额外安装）
══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
import os
import sys

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════
# DataSchema —— 真实数据 CSV 列名说明
# ════════════════════════════════════════════════════════════
"""
date                  : 日期，YYYY-MM-01，月度频率
price                 : 罗非鱼出塘价（元/公斤）

--- 先行指标 ---
us_inventory          : 美国收购商仓库库存指数（基准100，高→承压）
cn_factory_inventory  : 中国加工厂仓库库存指数（基准100，高→下压）
pond_stock_level      : 鱼塘存塘量指数（基准100，高→供应过剩）
fry_release_volume    : 当月苗种放苗量指数（基准100）
                        ⚠ 注：4个月后的价格与此相关性最高
feed_sales_volume     : 主流饲料厂罗非鱼饲料销量指数（基准100）
                        ⚠ 注：同样有4个月先行效应

--- 外生因子 ---
soybean_price         : 豆粕价格（元/吨）
corn_price            : 玉米价格（元/吨）
weather_risk          : 天气风险（0=正常 1=中风险 2=高风险/灾害）
export_index          : 出口量指数（基准100）
"""


# ════════════════════════════════════════════════════════════
# 1. 高保真数据生成器
# ════════════════════════════════════════════════════════════

class TilapiaDataGenerator:
    """
    生成符合行业逻辑的模拟月度历史数据。
    内含全部9类因子的协变量，并确保各变量之间的因果关系正确。
    """

    @staticmethod
    def generate(start='2018-01-01', end='2025-12-31', freq='MS') -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end, freq=freq)
        n = len(dates)
        np.random.seed(42)
        months = np.array([d.month for d in dates])
        years  = np.array([d.year  for d in dates])
        t      = np.arange(n)

        # ── 先行指标生成 ──────────────────────────────────────

        # 4. 放苗量（苗种），季节性：3-4月投苗旺季（华南），6月次旺季
        fry_seasonal = np.array([
            60, 50, 130, 140, 90, 110, 70, 65, 70, 80, 70, 55
        ])
        fry_base = fry_seasonal[months - 1].astype(float)
        fry = fry_base + np.random.normal(0, 10, n)
        fry += t * 0.3   # 行业缓慢扩张
        fry = np.clip(fry, 30, 200)

        # 5. 饲料销量（与放苗量高度相关，延迟约1个月）
        feed_sales = np.roll(fry, 1) * 1.1 + np.random.normal(0, 8, n)
        feed_sales = np.clip(feed_sales, 40, 220)

        # 3. 存塘量 = 过去4个月放苗量的积累，调整死亡率
        pond_stock = np.zeros(n)
        for i in range(4, n):
            # 累积4个月放苗量，模拟存塘（死亡率约15%/月叠加）
            pond_stock[i] = (fry[i-4]*0.85 + fry[i-3]*0.7 +
                             fry[i-2]*0.5  + fry[i-1]*0.3)
        pond_stock[:4] = pond_stock[4]  # 填充开头
        pond_stock = pond_stock / pond_stock.mean() * 100  # 归一化到100

        # 1. 美国仓库库存 —— 与中国出口量反向，有1-2月滞后
        #    库存堆积规律：通常年底消费旺季前（9-10月）备货，年初出清
        us_inv_seasonal = np.array([110, 108, 100, 95, 90, 88, 92, 100, 108, 112, 115, 115])
        us_inventory = us_inv_seasonal[months - 1].astype(float)
        us_inventory += np.random.normal(0, 5, n)
        us_inventory[years == 2019] += 15  # 贸易战：美国减少采购，库存积压
        us_inventory[years == 2020] -= 10  # 疫情：消费下降，但采购也减少
        us_inventory = np.clip(us_inventory, 70, 140)

        # 2. 中国加工厂库存 —— 与出塘旺季（7-9月）同步积累
        cn_inv_seasonal = np.array([95, 90, 88, 90, 95, 100, 115, 120, 118, 108, 100, 97])
        cn_factory_inv = cn_inv_seasonal[months - 1].astype(float)
        cn_factory_inv += np.random.normal(0, 6, n)
        cn_factory_inv[years == 2020] += 20  # 疫情：加工停滞，库存积压
        cn_factory_inv = np.clip(cn_factory_inv, 60, 150)

        # ── 外生因子 ──────────────────────────────────────────

        # 饲料成本
        soybean = 3200 + t * 8 + np.random.normal(0, 150, n)
        soybean[months <= 3] += 100
        corn    = 2400 + t * 5 + np.random.normal(0, 100, n)
        feed_cost_idx = 100 * (soybean * 0.3 + corn * 0.2) / (3200 * 0.3 + 2400 * 0.2)

        # 天气风险
        weather_risk = np.zeros(n, dtype=int)
        for i, d in enumerate(dates):
            if d.month in [7, 8, 9]: weather_risk[i] = 1
            if d.year == 2022 and d.month in [1, 2]: weather_risk[i] = 2
            if d.year == 2020 and d.month == 8: weather_risk[i] = 2  # 洪涝

        # 出口指数
        export_idx = 100 - t * 0.3 + np.random.normal(0, 3, n)
        export_idx[years == 2019] -= 15
        export_idx[years == 2020] -= 20
        export_idx = np.clip(export_idx, 50, 120)

        # ── 价格生成（基于各因子的实际因果效应）────────────────

        # 基础趋势
        base_price = 10.0 * (1.015 ** (t / 12))

        # 季节效应
        seasonal = -0.8 * np.cos(2*np.pi*(months-1)/12) + 0.3*np.cos(4*np.pi*(months-1)/12)

        # 先行指标对价格的影响（4个月后效应）
        fry_4m_lag = np.roll(fry, 4)
        fry_4m_lag[:4] = fry_4m_lag[4]
        fry_price_effect = -(fry_4m_lag - fry_4m_lag.mean()) / fry_4m_lag.std() * 0.5

        feed_sales_4m_lag = np.roll(feed_sales, 4)
        feed_sales_4m_lag[:4] = feed_sales_4m_lag[4]
        feed_price_effect = -(feed_sales_4m_lag - feed_sales_4m_lag.mean()) / feed_sales_4m_lag.std() * 0.3

        pond_effect = -(pond_stock - 100) / 20 * 0.4
        us_inv_effect = -(us_inventory - 100) / 15 * 0.6
        cn_inv_effect = -(cn_factory_inv - 100) / 15 * 0.5
        feed_cost_effect = (feed_cost_idx - 100) / 10 * 0.25

        # 天气冲击
        shock = np.zeros(n)
        for i, d in enumerate(dates):
            if d.year == 2019 and d.month >= 7: shock[i] = -0.5
            if d.year == 2020 and d.month in [2,3,4]: shock[i] = -1.0
            if d.year == 2020 and d.month in [8,9]: shock[i] = 0.6
            if d.year == 2022 and d.month in [1,2]: shock[i] = 2.0
            if weather_risk[i] == 2: shock[i] += 0.8

        noise = np.random.normal(0, 0.3, n)

        price = (base_price + seasonal + fry_price_effect + feed_price_effect +
                 pond_effect + us_inv_effect + cn_inv_effect + feed_cost_effect +
                 shock + noise)
        price = np.clip(price, 7.0, 20.0)

        df = pd.DataFrame({
            'date':                dates,
            'price':               price.round(2),
            # 先行指标
            'fry_release_volume':  fry.round(1),
            'feed_sales_volume':   feed_sales.round(1),
            'pond_stock_level':    pond_stock.round(1),
            'us_inventory':        us_inventory.round(1),
            'cn_factory_inventory':cn_factory_inv.round(1),
            # 外生因子
            'soybean_price':       soybean.round(0).astype(int),
            'corn_price':          corn.round(0).astype(int),
            'feed_cost_index':     feed_cost_idx.round(2),
            'weather_risk':        weather_risk,
            'export_index':        export_idx.round(1),
            # 时间辅助
            'month': months,
            'year':  years,
        })
        df.set_index('date', inplace=True)
        return df

    @staticmethod
    def load_from_csv(path: str) -> pd.DataFrame:
        """
        从CSV加载真实数据，必须包含 DataSchema 中的列（price为必填）。
        其余列缺失时将用均值填充并给出警告。
        """
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        df['month'] = df.index.month
        df['year']  = df.index.year

        REQUIRED = ['price']
        OPTIONAL = ['fry_release_volume','feed_sales_volume','pond_stock_level',
                    'us_inventory','cn_factory_inventory','soybean_price',
                    'corn_price','feed_cost_index','weather_risk','export_index']
        for col in REQUIRED:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要列: {col}")
        for col in OPTIONAL:
            if col not in df.columns:
                df[col] = 100.0
                print(f"  [⚠] 列 '{col}' 缺失，已用默认值100填充，预测精度可能下降")
        return df


# ════════════════════════════════════════════════════════════
# 2. 特征工程
# ════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    构建完整特征矩阵，涵盖：
    - 自回归特征（滞后价格）
    - 季节性特征（sin/cos编码）
    - 先行指标（放苗/饲料销量的4个月滞后 = 当前的先行信号）
    - 库存指标（美国仓库、加工厂仓库）
    - 存塘量
    - 成本指标（饲料成本）
    - 天气风险
    """

    @staticmethod
    def build_features(df: pd.DataFrame, price_lags: int = 3) -> pd.DataFrame:
        feat = df.copy()

        # ── 自回归特征 ──
        for lag in range(1, price_lags + 1):
            feat[f'price_lag_{lag}'] = feat['price'].shift(lag)
        feat['price_ma3'] = feat['price'].shift(1).rolling(3).mean()
        feat['price_ma6'] = feat['price'].shift(1).rolling(6).mean()

        # ── 季节性编码 ──
        feat['month_sin'] = np.sin(2 * np.pi * feat['month'] / 12)
        feat['month_cos'] = np.cos(2 * np.pi * feat['month'] / 12)
        quarter = ((feat['month'] - 1) // 3) + 1
        for q in [1, 2, 3, 4]:
            feat[f'q{q}'] = (quarter == q).astype(int)

        # ── 先行指标：放苗量（4个月前的放苗 = 当前供应信号）──
        # 4个月前放了多少苗，决定了当前出塘量
        feat['fry_4m_ago']       = feat['fry_release_volume'].shift(4)
        feat['fry_3m_ago']       = feat['fry_release_volume'].shift(3)
        feat['feed_sales_4m_ago']= feat['feed_sales_volume'].shift(4)
        # 放苗量变化率
        feat['fry_change_4m']    = feat['fry_release_volume'].shift(4).pct_change(3)

        # ── 当前库存压力 ──
        feat['us_inv_norm']      = (feat['us_inventory'] - 100) / 20
        feat['cn_inv_norm']      = (feat['cn_factory_inventory'] - 100) / 20
        feat['pond_stock_norm']  = (feat['pond_stock_level'] - 100) / 20
        # 综合库存压力指数（库存越高，价格压力越大）
        feat['total_inventory_pressure'] = (
            feat['us_inv_norm'] * 0.4 +
            feat['cn_inv_norm'] * 0.35 +
            feat['pond_stock_norm'] * 0.25
        )

        # ── 饲料成本 ──
        feat['feed_cost_norm']   = (feat['feed_cost_index'] - 100) / 10
        feat['feed_cost_change'] = feat['feed_cost_index'].pct_change()

        # ── 天气风险 ──
        feat['high_weather']     = (feat['weather_risk'] == 2).astype(int)
        feat['mid_weather']      = (feat['weather_risk'] == 1).astype(int)

        # ── 出口 ──
        feat['export_norm']      = (feat['export_index'] - 100) / 20

        # ── 时间趋势 ──
        feat['time_idx']         = np.arange(len(feat))

        feat.dropna(inplace=True)
        return feat


# ════════════════════════════════════════════════════════════
# 3. 纯 NumPy 模型实现
# ════════════════════════════════════════════════════════════

class ExponentialSmoothing:
    """双指数平滑（Holt法），捕捉趋势与惯性"""

    def __init__(self, alpha=0.25, beta=0.05):
        self.alpha, self.beta = alpha, beta
        self.level_ = self.trend_ = None

    def fit(self, series):
        a, b = self.alpha, self.beta
        n = len(series)
        L, T = np.zeros(n), np.zeros(n)
        L[0] = series[0]
        T[0] = series[1] - series[0] if n > 1 else 0
        for t in range(1, n):
            L[t] = a * series[t] + (1-a) * (L[t-1] + T[t-1])
            T[t] = b * (L[t] - L[t-1]) + (1-b) * T[t-1]
        self.level_, self.trend_ = L[-1], T[-1]
        return self

    def predict(self, steps):
        return np.array([self.level_ + h * self.trend_ for h in range(1, steps+1)])


class RidgeRegression:
    """L2正则化线性回归（Ridge），防止过拟合"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = self.intercept_ = None

    def fit(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]
        k = Xb.shape[1]
        A = Xb.T @ Xb + self.alpha * np.eye(k)
        A[0, 0] -= self.alpha
        self.coef_all_ = np.linalg.solve(A, Xb.T @ y)
        self.intercept_ = self.coef_all_[0]
        self.coef_ = self.coef_all_[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class SeasonalModel:
    """季节性 + 趋势线性模型"""

    def __init__(self):
        self.lr = RidgeRegression(alpha=0.01)
        self.cols = ['time_idx', 'month_sin', 'month_cos', 'q1', 'q2', 'q3', 'q4']

    def fit(self, df):
        self.lr.fit(df[self.cols].values, df['price'].values)
        return self

    def predict(self, df):
        return self.lr.predict(df[self.cols].values)


class MultiFactorModel:
    """
    多因子岭回归模型，整合全部9类因子。
    特征包含：
      自回归、季节性、先行指标（放苗/饲料销量4月滞后）、
      综合库存压力、饲料成本、天气风险、出口、趋势
    """

    # 核心特征列表
    FEATURES = [
        # 自回归
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'price_ma3', 'price_ma6',
        # 季节性
        'month_sin', 'month_cos',
        # ★ 先行指标（放苗/饲料销量4月滞后）
        'fry_4m_ago', 'fry_3m_ago', 'feed_sales_4m_ago', 'fry_change_4m',
        # ★ 库存压力（美国仓库 + 加工厂 + 存塘量）
        'total_inventory_pressure', 'us_inv_norm', 'cn_inv_norm', 'pond_stock_norm',
        # 成本
        'feed_cost_norm', 'feed_cost_change',
        # 天气
        'high_weather', 'mid_weather',
        # 出口
        'export_norm',
        # 趋势
        'time_idx',
    ]

    def __init__(self):
        self.lr = RidgeRegression(alpha=2.0)
        self.mean_ = self.std_ = None

    def fit(self, df):
        X = df[self.FEATURES].values.astype(float)
        y = df['price'].values
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        self.lr.fit((X - self.mean_) / self.std_, y)
        return self

    def predict(self, df):
        X = df[self.FEATURES].values.astype(float)
        return self.lr.predict((X - self.mean_) / self.std_)

    def feature_importance(self):
        """返回各特征的重要性（绝对标准化系数）"""
        return dict(zip(self.FEATURES, self.lr.coef_))


# ════════════════════════════════════════════════════════════
# 4. 集成预测器
# ════════════════════════════════════════════════════════════

class TilapiaPriceEnsemble:
    """
    三模型加权集成：
      ETS（短期惯性）+ 季节模型（周期性）+ 多因子模型（全信息）
    权重由验证集MAE自动学习。
    """

    def __init__(self):
        self.ets = ExponentialSmoothing(alpha=0.25, beta=0.05)
        self.seasonal = SeasonalModel()
        self.multi = MultiFactorModel()
        self.weights = np.array([0.15, 0.25, 0.60])
        self.feat_df = None
        self.trained = False

    def fit(self, df, val_ratio=0.15):
        feat_df = FeatureEngineer.build_features(df)
        n_val = max(4, int(len(feat_df) * val_ratio))
        train, val = feat_df.iloc[:-n_val], feat_df.iloc[-n_val:]

        self.ets.fit(train['price'].values)
        self.seasonal.fit(train)
        self.multi.fit(train)

        vy = val['price'].values
        preds = np.column_stack([
            self.ets.predict(len(val)),
            self.seasonal.predict(val),
            self.multi.predict(val),
        ])
        maes = np.mean(np.abs(preds - vy[:, None]), axis=0)
        inv_mae = 1 / (maes + 1e-6)
        self.weights = inv_mae / inv_mae.sum()

        # 全量重训
        self.ets.fit(feat_df['price'].values)
        self.seasonal.fit(feat_df)
        self.multi.fit(feat_df)

        self.feat_df = feat_df
        self.raw_df = df
        self.trained = True

        return {
            'ETS MAE':      f"{maes[0]:.3f} 元/kg",
            '季节模型 MAE':  f"{maes[1]:.3f} 元/kg",
            '多因子模型 MAE':f"{maes[2]:.3f} 元/kg",
            '集成权重':      (f"ETS={self.weights[0]:.2f}  "
                              f"季节={self.weights[1]:.2f}  "
                              f"多因子={self.weights[2]:.2f}"),
        }

    def predict_next_n_months(self, n=6, extra_factors=None):
        """
        滚动预测未来n个月。

        extra_factors: dict，可提供未来各月的外生因子，例如：
        {
          'fry_release_volume':   [120,130,...],   # 本月放苗量（已知当月数据时提供）
          'feed_sales_volume':    [115,120,...],
          'us_inventory':         [105,110,...],
          'cn_factory_inventory': [115,112,...],
          'pond_stock_level':     [108,110,...],
          'feed_cost_index':      [102,104,...],
          'weather_risk':         [0,0,1,2,1,0],
          'export_index':         [82,81,...],
        }
        """
        if not self.trained:
            raise RuntimeError("请先调用 fit()")

        fd = self.feat_df.copy()
        history_raw = list(self.raw_df.to_dict('records'))
        history_prices = list(fd['price'].values)
        last_tidx = int(fd['time_idx'].values[-1])

        future_dates = pd.date_range(
            start=fd.index[-1] + pd.DateOffset(months=1),
            periods=n, freq='MS')

        future_rows = []
        for i, dt in enumerate(future_dates):
            m = dt.month
            q = (m - 1) // 3 + 1
            last_raw = history_raw[-1]

            def ef(key, default, idx=i):
                if extra_factors and key in extra_factors:
                    v = extra_factors[key]
                    return float(v[idx]) if idx < len(v) else float(v[-1])
                return default

            # 先行指标默认值（无额外输入时，延续近期趋势）
            fry_now     = ef('fry_release_volume',   last_raw.get('fry_release_volume', 100) * 1.005)
            feed_sales_now = ef('feed_sales_volume', last_raw.get('feed_sales_volume', 100) * 1.005)
            us_inv      = ef('us_inventory',         last_raw.get('us_inventory', 100) * (1 + 0.002*(m in [9,10,11] and 1 or -1)))
            cn_inv      = ef('cn_factory_inventory', last_raw.get('cn_factory_inventory', 100) * (1 + 0.003*(m in [7,8,9] and 1 or -1)))
            pond_stock  = ef('pond_stock_level',     last_raw.get('pond_stock_level', 100))
            feed_idx    = ef('feed_cost_index',      last_raw.get('feed_cost_index', 100) * 1.003)
            w_risk      = int(ef('weather_risk',     1 if m in [7,8,9] else 0))
            exp_idx     = ef('export_index',         max(50, last_raw.get('export_index', 90) - 0.3))
            soy         = last_raw.get('soybean_price', 3500) * 1.002
            corn        = last_raw.get('corn_price', 2600) * 1.001

            hp = history_prices
            lag1 = hp[-1]; lag2 = hp[-2] if len(hp)>=2 else lag1
            lag3 = hp[-3] if len(hp)>=3 else lag2
            ma3  = np.mean(hp[-3:]);  ma6 = np.mean(hp[-6:]) if len(hp)>=6 else np.mean(hp)

            # 重建历史raw队列以计算4月滞后
            raw_fry  = [r.get('fry_release_volume', 100) for r in history_raw]
            raw_feed = [r.get('feed_sales_volume',  100) for r in history_raw]
            fry_4m   = raw_fry[-4] if len(raw_fry) >= 4 else 100
            fry_3m   = raw_fry[-3] if len(raw_fry) >= 3 else 100
            feed_4m  = raw_feed[-4] if len(raw_feed) >= 4 else 100
            fry_chg  = (fry_4m - (raw_fry[-7] if len(raw_fry)>=7 else fry_4m)) / (abs(raw_fry[-7]) + 1e-6) if len(raw_fry)>=7 else 0

            fc_prev = last_raw.get('feed_cost_index', 100)
            feed_change = (feed_idx - fc_prev) / (fc_prev + 1e-6)

            us_n  = (us_inv - 100) / 20
            cn_n  = (cn_inv - 100) / 20
            ps_n  = (pond_stock - 100) / 20
            total_press = us_n*0.4 + cn_n*0.35 + ps_n*0.25

            row = {
                'price':                   None,
                'fry_release_volume':      fry_now,
                'feed_sales_volume':       feed_sales_now,
                'pond_stock_level':        pond_stock,
                'us_inventory':            us_inv,
                'cn_factory_inventory':    cn_inv,
                'soybean_price':           soy,
                'corn_price':              corn,
                'feed_cost_index':         feed_idx,
                'weather_risk':            w_risk,
                'export_index':            exp_idx,
                'month': m, 'year': dt.year,
                # features
                'price_lag_1': lag1, 'price_lag_2': lag2, 'price_lag_3': lag3,
                'price_ma3': ma3, 'price_ma6': ma6,
                'month_sin': np.sin(2*np.pi*m/12),
                'month_cos': np.cos(2*np.pi*m/12),
                'q1':int(q==1),'q2':int(q==2),'q3':int(q==3),'q4':int(q==4),
                'fry_4m_ago':        fry_4m,
                'fry_3m_ago':        fry_3m,
                'feed_sales_4m_ago': feed_4m,
                'fry_change_4m':     fry_chg,
                'total_inventory_pressure': total_press,
                'us_inv_norm': us_n, 'cn_inv_norm': cn_n, 'pond_stock_norm': ps_n,
                'feed_cost_norm':  (feed_idx - 100) / 10,
                'feed_cost_change': feed_change,
                'high_weather': int(w_risk==2),
                'mid_weather':  int(w_risk==1),
                'export_norm':  (exp_idx - 100) / 20,
                'time_idx':     last_tidx + i + 1,
            }

            tmp = pd.DataFrame([row])
            tmp.index = pd.DatetimeIndex([dt])

            ets_p  = float(self.ets.predict(i+1)[-1])
            sea_p  = float(self.seasonal.predict(tmp)[0])
            mfm_p  = float(self.multi.predict(tmp)[0])
            ens_p  = float(np.dot([ets_p, sea_p, mfm_p], self.weights))
            ens_p  = max(7.0, min(20.0, ens_p))

            row['price'] = ens_p
            future_rows.append(row)
            history_prices.append(ens_p)
            history_raw.append(row)

        # 批量预测（更稳定）
        fut_df = pd.DataFrame(future_rows)
        fut_df.index = pd.DatetimeIndex(future_dates)

        ets_preds = self.ets.predict(n)
        sea_preds = self.seasonal.predict(fut_df)
        mfm_preds = self.multi.predict(fut_df)
        ensemble  = np.dot(np.column_stack([ets_preds, sea_preds, mfm_preds]), self.weights)
        ensemble  = np.clip(ensemble, 7.0, 20.0)

        # 预测区间（基于训练残差）
        train_res = self.feat_df['price'].values - self.multi.predict(self.feat_df)
        sigma = train_res.std()
        lower = np.clip(ensemble - 1.96*sigma, 7.0, 20.0)
        upper = np.clip(ensemble + 1.96*sigma, 7.0, 20.0)

        result = pd.DataFrame({
            '预测月份':         future_dates,
            '预测价格(元/kg)':  ensemble.round(2),
            '下界(95%CI)':      lower.round(2),
            '上界(95%CI)':      upper.round(2),
            'ETS预测':          ets_preds.round(2),
            '季节模型':         sea_preds.round(2),
            '多因子模型':       mfm_preds.round(2),
            '库存压力':         fut_df['total_inventory_pressure'].round(2).values,
            '放苗(4M前)':       fut_df['fry_4m_ago'].round(1).values,
        })
        result['价格趋势'] = result['预测价格(元/kg)'].diff().apply(
            lambda x: '↑ 上涨' if x>0.1 else ('↓ 下跌' if x<-0.1 else '→ 平稳'))
        result.iloc[0, result.columns.get_loc('价格趋势')] = '—'
        return result


# ════════════════════════════════════════════════════════════
# 5. 可视化
# ════════════════════════════════════════════════════════════

class Visualizer:

    @staticmethod
    def plot_dashboard(hist_df, forecast_df, save_path=None):
        """四格综合仪表板"""
        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
        fig.suptitle('Tilapia Price Prediction Dashboard\n罗非鱼鱼价预测综合仪表板',
                     fontsize=14, fontweight='bold', y=0.99)

        # ── 图1：历史 + 预测 ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        h = hist_df.iloc[-30:]
        ax1.plot(h.index, h['price'], 'o-', color='#2196F3',
                 lw=2, ms=4, label='Historical / 历史价格')

        fc_d = forecast_df['预测月份']
        fc_p = forecast_df['预测价格(元/kg)'].values
        ax1.plot(fc_d, fc_p, 's--', color='#F44336', lw=2.5, ms=6, label='Forecast / 预测', zorder=5)
        ax1.fill_between(fc_d,
                         forecast_df['下界(95%CI)'].values,
                         forecast_df['上界(95%CI)'].values,
                         color='#F44336', alpha=0.12, label='95% CI')

        ax1.axvline(hist_df.index[-1], color='gray', ls=':', lw=1.5, alpha=0.7)
        ax1.set_ylabel('Price (CNY/kg)', fontsize=10)
        ax1.set_title('Price History & Forecast / 历史价格与预测（近30月+未来6月）', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        # ── 图2：季节性 ──────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        monthly = hist_df.groupby('month')['price'].mean()
        colors  = ['#FF5722' if m in [7,8,9] else
                   '#4CAF50' if m in [11,12,1,2] else '#90CAF9'
                   for m in monthly.index]
        ax2.bar(monthly.index, monthly.values, color=colors, alpha=0.85)
        ax2.set_xticks(range(1,13))
        ax2.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax2.set_title('Seasonal Pattern / 季节性规律', fontsize=11)
        ax2.set_ylabel('Avg Price (CNY/kg)', fontsize=9)
        ax2.grid(True, axis='y', alpha=0.3)

        # ── 图3：库存压力 vs 价格 ──────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        feat_df = FeatureEngineer.build_features(hist_df)
        ax3.scatter(feat_df['total_inventory_pressure'], feat_df['price'],
                    alpha=0.5, color='#7986CB', s=25)
        # 线性拟合
        x = feat_df['total_inventory_pressure'].values
        y = feat_df['price'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            coef = np.polyfit(x[mask], y[mask], 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax3.plot(xline, np.polyval(coef, xline), 'r-', lw=2, alpha=0.8)
        ax3.set_xlabel('Inventory Pressure Index', fontsize=9)
        ax3.set_ylabel('Price (CNY/kg)', fontsize=9)
        ax3.set_title('Inventory Pressure vs Price\n库存压力 vs 鱼价（负相关）', fontsize=11)
        ax3.grid(True, alpha=0.3)

        path = save_path or os.path.join(OUTPUT_DIR, 'tilapia_dashboard.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    @staticmethod
    def plot_leading_indicators(hist_df, save_path=None):
        """先行指标与价格的关系（放苗量4月滞后）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Leading Indicators Analysis / 先行指标分析',
                     fontsize=13, fontweight='bold')

        indicators = [
            ('fry_release_volume',    '放苗量指数（当月）',    4, '#4CAF50'),
            ('feed_sales_volume',     '饲料销量指数（当月）',  4, '#FF9800'),
            ('us_inventory',          '美国仓库库存指数',      0, '#E91E63'),
            ('cn_factory_inventory',  '中国加工厂库存指数',    0, '#3F51B5'),
        ]

        for ax, (col, title, lead, color) in zip(axes.flat, indicators):
            series = hist_df[col].shift(lead) if lead > 0 else hist_df[col]
            price  = hist_df['price']
            valid  = series.notna() & price.notna()
            x, y   = series[valid].values, price[valid].values

            ax.scatter(x, y, alpha=0.45, color=color, s=20)
            if len(x) > 2:
                coef = np.polyfit(x, y, 1)
                xl   = np.linspace(x.min(), x.max(), 100)
                ax.plot(xl, np.polyval(coef, xl), 'k-', lw=2, alpha=0.8)
                corr = np.corrcoef(x, y)[0, 1]
                ax.set_title(f"{title}\n(vs Price, lag={lead}M, r={corr:.2f})", fontsize=10)
            ax.set_ylabel('Price (CNY/kg)', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or os.path.join(OUTPUT_DIR, 'tilapia_leading_indicators.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    @staticmethod
    def plot_feature_importance(model, save_path=None):
        fi   = model.feature_importance()
        keys = [k for k, v in sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)]
        vals = [fi[k] for k in keys]
        colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]

        # 友好的中文标签映射
        label_map = {
            'price_lag_1': '上月价格(滞后1月)',
            'price_lag_2': '滞后2月价格',
            'price_lag_3': '滞后3月价格',
            'price_ma3':   '3月均价',
            'price_ma6':   '6月均价',
            'month_sin':   '季节-Sin',
            'month_cos':   '季节-Cos',
            'fry_4m_ago':          '★放苗量(4月前)',
            'fry_3m_ago':          '放苗量(3月前)',
            'feed_sales_4m_ago':   '★饲料销量(4月前)',
            'fry_change_4m':       '放苗量变化率',
            'total_inventory_pressure': '★综合库存压力',
            'us_inv_norm':         '★美国仓库库存',
            'cn_inv_norm':         '★加工厂库存',
            'pond_stock_norm':     '★鱼塘存塘量',
            'feed_cost_norm':      '饲料成本',
            'feed_cost_change':    '饲料成本变化率',
            'high_weather':        '高风险天气',
            'mid_weather':         '中风险天气',
            'export_norm':         '出口指数',
            'time_idx':            '时间趋势',
        }
        labels = [label_map.get(k, k) for k in keys]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(keys)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title('Feature Importance (Standardized Coef.)\n多因子模型特征重要性（★标注为行业经验先行指标）',
                     fontsize=11)
        ax.set_xlabel('Coefficient Value', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)

        path = save_path or os.path.join(OUTPUT_DIR, 'tilapia_feature_importance.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path


# ════════════════════════════════════════════════════════════
# 6. Agent 交互界面
# ════════════════════════════════════════════════════════════

class TilapiaPriceAgent:
    """
    罗非鱼鱼价预测 Agent
    ─────────────────────────────────────────────
    整合了行业经验的5个核心先行指标：
      1. 美国收购商仓库库存
      2. 中国加工厂仓库库存
      3. 鱼塘存塘量
      4. 苗种放苗量（T-4月先行指标）
      5. 主流饲料厂饲料销量（T-4月先行指标）
    ─────────────────────────────────────────────
    """

    HELP = """
╔═══════════════════════════════════════════════════════════════╗
║         罗非鱼鱼价预测 Agent v2.0 —— 命令列表                  ║
╠═══════════════════════════════════════════════════════════════╣
║  predict [N]      预测未来N个月价格（默认6个月）                ║
║  scenario         情景分析（自定义5个先行指标 + 外生因子）       ║
║  seasonal         季节性规律分析                               ║
║  factors          各因子影响分析（含先行指标相关性）             ║
║  leading          先行指标深度分析                             ║
║  chart            生成综合仪表板图表                           ║
║  history          查看近12月历史数据                           ║
║  advice           获取养殖经营建议                             ║
║  update_data      更新某月的实际数据                           ║
║  help             显示帮助                                    ║
║  quit             退出                                        ║
╚═══════════════════════════════════════════════════════════════╝
"""

    def __init__(self, data_path=None):
        print("\n" + "═"*62)
        print("  🐟  罗非鱼鱼价预测 Agent v2.0  正在启动")
        print("═"*62)

        if data_path and os.path.exists(data_path):
            print(f"[数据] 加载真实数据：{data_path}")
            self.df = TilapiaDataGenerator.load_from_csv(data_path)
        else:
            print("[数据] 使用模拟历史数据（2018-2025年，含5类先行指标）")
            self.df = TilapiaDataGenerator.generate()

        print("[模型] 训练中（含先行指标：放苗量、饲料销量、库存压力）...")
        self.ensemble = TilapiaPriceEnsemble()
        metrics = self.ensemble.fit(self.df)

        print("\n[验证集性能]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        print(f"\n[就绪] {len(self.df)} 个月数据，"
              f"{self.df.index[0].strftime('%Y-%m')} ~ {self.df.index[-1].strftime('%Y-%m')}")
        print(self.HELP)

    def run(self):
        while True:
            try:
                cmd = input("命令 > ").strip()
                if not cmd:
                    continue
                self._dispatch(cmd)
            except (KeyboardInterrupt, EOFError):
                print("\n\n再见！祝养殖顺利 🐟")
                break

    def _dispatch(self, cmd):
        p = cmd.lower().split()
        a = p[0] if p else ''

        if a in ('quit','exit','q'):
            raise SystemExit
        elif a == 'help':
            print(self.HELP)
        elif a == 'predict':
            self._predict(int(p[1]) if len(p)>1 else 6)
        elif a == 'scenario':
            self._scenario()
        elif a == 'seasonal':
            self._seasonal()
        elif a == 'factors':
            self._factors()
        elif a == 'leading':
            self._leading()
        elif a == 'chart':
            self._chart()
        elif a == 'history':
            self._history()
        elif a == 'advice':
            self._advice()
        elif a == 'update_data':
            self._update_data()
        else:
            self._nlp(cmd)

    # ── 命令实现 ──────────────────────────────────────────

    def _predict(self, n=6):
        print(f"\n📈 预测未来 {n} 个月...\n")
        r = self.ensemble.predict_next_n_months(n)
        print(f"{'月份':<10} {'预测价格':>10} {'下界':>7} {'上界':>7} "
              f"{'库存压力':>8} {'放苗(4M前)':>10} {'趋势':>8}")
        print("─"*70)
        for _, row in r.iterrows():
            print(f"  {row['预测月份'].strftime('%Y-%m'):<8} "
                  f"{row['预测价格(元/kg)']:>8.2f}元 "
                  f"{row['下界(95%CI)']:>6.2f} "
                  f"{row['上界(95%CI)']:>6.2f}  "
                  f"{row['库存压力']:>+8.2f}  "
                  f"{row['放苗(4M前)']:>9.1f}  "
                  f"  {row['价格趋势']}")
        print("─"*70)
        avg = r['预测价格(元/kg)'].mean()
        pk  = r.loc[r['预测价格(元/kg)'].idxmax()]
        lw  = r.loc[r['预测价格(元/kg)'].idxmin()]
        print(f"\n  均价预测：{avg:.2f} 元/kg")
        print(f"  高点：{pk['预测月份'].strftime('%Y-%m')}  {pk['预测价格(元/kg)']:.2f} 元/kg")
        print(f"  低点：{lw['预测月份'].strftime('%Y-%m')}  {lw['预测价格(元/kg)']:.2f} 元/kg")
        print("\n  [注] 库存压力>0 表示库存偏高（价格承压），<0 表示库存偏低（价格支撑）")

    def _scenario(self):
        print("\n🔧 情景分析 —— 请输入未来6个月各指标值")
        print("  直接回车使用默认值（基于近期趋势外推）\n")
        n = 6

        def ask(prompt, default):
            raw = input(f"  {prompt} [默认={default}，逗号分隔{n}个]: ").strip()
            if not raw:
                return [default] * n
            vals = [float(x.strip()) for x in raw.split(',')]
            return (vals + [vals[-1]] * n)[:n]

        print("  === 先行指标 ===")
        us_inv  = ask("美国仓库库存指数（基准100，高=压价）", 105)
        cn_inv  = ask("中国加工厂库存指数（基准100，高=压价）", 110)
        pond    = ask("鱼塘存塘量指数（基准100，高=压价）", 105)
        fry     = ask("本月放苗量指数（基准100）", 100)
        feed_s  = ask("饲料厂饲料销量指数（基准100）", 100)
        print("  === 外生因子 ===")
        feed_c  = ask("饲料成本指数（基准100）", 103)
        weather = ask("天气风险（0=正常,1=中,2=高）", 0)
        export  = ask("出口指数（基准100）", 82)

        extra = {
            'us_inventory':         us_inv,
            'cn_factory_inventory': cn_inv,
            'pond_stock_level':     pond,
            'fry_release_volume':   fry,
            'feed_sales_volume':    feed_s,
            'feed_cost_index':      feed_c,
            'weather_risk':         [int(w) for w in weather],
            'export_index':         export,
        }

        r_scene = self.ensemble.predict_next_n_months(n, extra_factors=extra)
        r_base  = self.ensemble.predict_next_n_months(n)

        print("\n" + "─"*60)
        print(f"{'月份':<10} {'情景价格':>10} {'基准价格':>10} {'差异':>10}")
        print("─"*60)
        for i, row in r_scene.iterrows():
            bp   = r_base.iloc[r_scene.index.get_loc(i)]['预测价格(元/kg)']
            diff = row['预测价格(元/kg)'] - bp
            print(f"  {row['预测月份'].strftime('%Y-%m'):<8} "
                  f"{row['预测价格(元/kg)']:>8.2f}元 "
                  f"{bp:>8.2f}元  "
                  f"  {'+' if diff>=0 else ''}{diff:.2f}元")
        print("─"*60)
        avg_diff = r_scene['预测价格(元/kg)'].mean() - r_base['预测价格(元/kg)'].mean()
        print(f"\n  情景 vs 基准均价差异：{'+' if avg_diff>=0 else ''}{avg_diff:.2f} 元/kg")

    def _seasonal(self):
        print("\n🌊 季节性规律\n")
        mo = self.df.groupby('month')['price'].agg(['mean','min','max','std'])
        gm = self.df['price'].mean()
        names = ['','一月','二月','三月','四月','五月','六月',
                 '七月','八月','九月','十月','十一月','十二月']
        print(f"  历史均价：{gm:.2f} 元/kg\n")
        print(f"  {'月份':<8} {'均价':>7} {'最低':>7} {'最高':>7} {'偏均值':>8}  特征")
        print("  " + "─"*58)
        for m, row in mo.iterrows():
            d = (row['mean'] - gm) / gm * 100
            if m in [7,8,9]:    tag = '🔴 集中出塘低价季'
            elif m in [11,12,1,2]: tag = '🟢 供应收紧高价季'
            elif m in [3,4]:    tag = '🌱 投苗旺季（4月后供应预警）'
            elif m in [9,10]:   tag = '📦 美国备货旺季（关注US库存）'
            else:               tag = '⬜ 平季'
            print(f"  {names[m]:<6} {row['mean']:>7.2f} {row['min']:>6.2f} "
                  f"{row['max']:>6.2f}  {('+' if d>=0 else '')}{d:>4.1f}%   {tag}")
        print()
        print("  💡 关键规律：")
        print("     • 3-4月投苗旺季 → 7-8月出塘集中 → 夏季低价")
        print("     • 饲料销量高峰（3-4月）可提前4个月预判夏季供应压力")
        print("     • 9-10月美国圣诞备货 → 加工厂积极收购 → 价格支撑")
        print("     • 冬季（11-2月）存塘减少 + 节日需求 → 价格偏高")

    def _leading(self):
        print("\n🔭 先行指标深度分析\n")
        df = self.df.copy()

        print("  [1] 放苗量 → 4个月后价格相关性")
        for lag in [2, 3, 4, 5]:
            shifted = df['fry_release_volume'].shift(lag)
            valid   = shifted.notna() & df['price'].notna()
            r       = np.corrcoef(shifted[valid], df['price'][valid])[0,1]
            bar     = '█' * int(abs(r)*20)
            print(f"      滞后{lag}月：r={r:+.3f}  {bar}")

        print("\n  [2] 饲料销量 → 4个月后价格相关性")
        for lag in [2, 3, 4, 5]:
            shifted = df['feed_sales_volume'].shift(lag)
            valid   = shifted.notna() & df['price'].notna()
            r       = np.corrcoef(shifted[valid], df['price'][valid])[0,1]
            bar     = '█' * int(abs(r)*20)
            print(f"      滞后{lag}月：r={r:+.3f}  {bar}")

        print("\n  [3] 美国仓库库存 → 同期价格相关性")
        valid = df['us_inventory'].notna() & df['price'].notna()
        r     = np.corrcoef(df['us_inventory'][valid], df['price'][valid])[0,1]
        print(f"      同期相关：r={r:+.3f}  （负相关：库存高→价格低）")

        print("\n  [4] 中国加工厂库存 → 同期价格相关性")
        valid = df['cn_factory_inventory'].notna() & df['price'].notna()
        r     = np.corrcoef(df['cn_factory_inventory'][valid], df['price'][valid])[0,1]
        print(f"      同期相关：r={r:+.3f}  （负相关：库存高→价格低）")

        print("\n  [5] 鱼塘存塘量 → 同期价格相关性")
        valid = df['pond_stock_level'].notna() & df['price'].notna()
        r     = np.corrcoef(df['pond_stock_level'][valid], df['price'][valid])[0,1]
        print(f"      同期相关：r={r:+.3f}  （负相关：存塘多→价格低）")

        print("\n  📌 实操意义：")
        print("     • 若本月放苗量/饲料销量异常放大 → 预警4个月后供应过剩")
        print("     • 若美国/加工厂库存持续累积   → 警惕近期收购价下压")
        print("     • 若存塘量偏高 + 美国库存高    → 双重压力，价格可能急跌")
        print("     • 若存塘量偏低 + 天气风险高    → 供应紧张，价格可能急涨")

    def _factors(self):
        print("\n🔍 各因子重要性分析\n")
        fi = self.ensemble.multi.feature_importance()
        label_map = {
            'price_lag_1':'上月价格',
            'fry_4m_ago':'★放苗量(4月前)',
            'feed_sales_4m_ago':'★饲料销量(4月前)',
            'total_inventory_pressure':'★综合库存压力',
            'us_inv_norm':'★美国仓库库存',
            'cn_inv_norm':'★加工厂库存',
            'pond_stock_norm':'★鱼塘存塘量',
            'month_sin':'季节性(sin)',
            'month_cos':'季节性(cos)',
            'feed_cost_norm':'饲料成本',
            'high_weather':'极端天气',
            'export_norm':'出口指数',
            'time_idx':'时间趋势',
        }
        top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
        print(f"  {'因子':<22} {'系数':>8}  影响方向")
        print("  " + "─"*50)
        for k, v in top:
            name = label_map.get(k, k)
            direction = '正向（↑）' if v > 0 else '负向（↓）'
            bar  = '█' * min(20, int(abs(v) * 8))
            print(f"  {name:<20} {v:>+8.3f}  {direction}  {bar}")
        print()
        print("  说明：★ 标注的为行业经验先行指标")
        print("  负系数 = 该指标越高 → 价格越低")
        print("  正系数 = 该指标越高 → 价格越高")

    def _chart(self):
        print("\n📊 生成图表...")
        fc   = self.ensemble.predict_next_n_months(6)
        p1   = Visualizer.plot_dashboard(self.df, fc)
        p2   = Visualizer.plot_leading_indicators(self.df)
        p3   = Visualizer.plot_feature_importance(self.ensemble.multi)
        print(f"  ✓ 综合仪表板：  {p1}")
        print(f"  ✓ 先行指标图：  {p2}")
        print(f"  ✓ 特征重要性：  {p3}")

    def _history(self):
        print("\n📋 近12月历史数据\n")
        h = self.df.iloc[-12:]
        print(f"  {'日期':<10} {'价格':>7} {'美国库存':>9} {'加厂库存':>9} "
              f"{'存塘量':>7} {'放苗量':>7} {'饲料销量':>9} {'天气'}")
        print("  " + "─"*72)
        for d, r in h.iterrows():
            rsk = {0:'正常', 1:'中风险', 2:'⚠高风险'}[int(r['weather_risk'])]
            print(f"  {d.strftime('%Y-%m'):<10} "
                  f"{r['price']:>6.2f}元 "
                  f"{r['us_inventory']:>8.1f} "
                  f"{r['cn_factory_inventory']:>8.1f} "
                  f"{r['pond_stock_level']:>7.1f} "
                  f"{r['fry_release_volume']:>7.1f} "
                  f"{r['feed_sales_volume']:>8.1f}  "
                  f"  {rsk}")
        print()

    def _advice(self):
        r          = self.ensemble.predict_next_n_months(6)
        avg_fc     = r['预测价格(元/kg)'].mean()
        last_price = self.df['price'].iloc[-1]
        last_row   = self.df.iloc[-1]
        max_m      = r.loc[r['预测价格(元/kg)'].idxmax(),'预测月份']
        min_m      = r.loc[r['预测价格(元/kg)'].idxmin(),'预测月份']
        trend      = '上涨' if avg_fc > last_price else '下跌'

        us_pressure  = 'HIGH ⚠' if last_row['us_inventory'] > 110 else 'OK'
        cn_pressure  = 'HIGH ⚠' if last_row['cn_factory_inventory'] > 115 else 'OK'
        pond_risk    = 'HIGH ⚠' if last_row['pond_stock_level'] > 110 else 'OK'

        print(f"""
💡 养殖经营建议（{datetime.now().strftime('%Y-%m-%d')} 生成）

  当前价格：{last_price:.2f} 元/kg
  未来6月均价预测：{avg_fc:.2f} 元/kg（预计{trend}）
  高点预计：{max_m.strftime('%Y-%m')}
  低点预计：{min_m.strftime('%Y-%m')}

  [当前市场信号]
  美国仓库库存：{last_row['us_inventory']:.0f}（{us_pressure}）
  中国加工厂库存：{last_row['cn_factory_inventory']:.0f}（{cn_pressure}）
  鱼塘存塘量：{last_row['pond_stock_level']:.0f}（{pond_risk}）

  [出塘策略]
  • {'⚠ 库存偏高，建议加快出塘，避免压塘损失' if 'HIGH' in (us_pressure+cn_pressure+pond_risk) else '✓ 库存正常，可按计划出塘'}
  • 高价窗口（{max_m.strftime('%Y-%m')}）前后可集中出塘
  • 低价期（{min_m.strftime('%Y-%m')}）尽量减少出塘量，可延迟至价格回升

  [放苗建议]
  • 大量放苗需考虑4个月后出塘时的价格区间
  • 当前放苗量高峰期 → 预警 {(self.df.index[-1]+pd.DateOffset(months=4)).strftime('%Y-%m')} 前后价格承压
  • 建议参考饲料厂销量趋势验证行业整体放苗规模

  [风险管理]
  • 持续监控美国/加工厂库存变动，库存超过120时果断加快出塘
  • 台风季（7-9月）关注天气预警，备好应急方案
  • 建议豆粕/玉米原料提前锁价，对冲成本上涨风险

  ⚠  以上仅为模型参考建议，请结合当地实际行情决策。
""")

    def _update_data(self):
        print("\n📝 更新月度数据（手动录入最新指标值）")
        print("  格式：YYYY-MM，回车确认\n")
        month_str = input("  输入月份（如 2025-12）: ").strip()
        try:
            dt = pd.to_datetime(month_str + '-01')
        except:
            print("  日期格式错误"); return

        price    = float(input("  实际出塘价（元/kg）: "))
        us_inv   = float(input("  美国仓库库存指数: "))
        cn_inv   = float(input("  加工厂库存指数: "))
        pond     = float(input("  存塘量指数: "))
        fry      = float(input("  放苗量指数: "))
        feed_s   = float(input("  饲料销量指数: "))
        feed_c   = float(input("  饲料成本指数: "))
        w_risk   = int(input("  天气风险（0/1/2）: "))
        exp_idx  = float(input("  出口指数: "))

        last = self.df.iloc[-1]
        new_row = {
            'price':                   price,
            'fry_release_volume':      fry,
            'feed_sales_volume':       feed_s,
            'pond_stock_level':        pond,
            'us_inventory':            us_inv,
            'cn_factory_inventory':    cn_inv,
            'soybean_price':           last['soybean_price'],
            'corn_price':              last['corn_price'],
            'feed_cost_index':         feed_c,
            'weather_risk':            w_risk,
            'export_index':            exp_idx,
            'month':                   dt.month,
            'year':                    dt.year,
        }
        new_df = pd.DataFrame([new_row], index=pd.DatetimeIndex([dt]))
        self.df = pd.concat([self.df, new_df])
        self.df = self.df[~self.df.index.duplicated(keep='last')]
        self.df.sort_index(inplace=True)

        print("\n  [模型] 使用更新后数据重新训练...")
        metrics = self.ensemble.fit(self.df)
        print("  [完成] 数据和模型已更新！")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    def _nlp(self, text):
        t = text.lower()
        if any(k in t for k in ['预测','价格','多少','未来']):
            import re; nums = re.findall(r'\d+', t)
            self._predict(int(nums[0]) if nums else 6)
        elif any(k in t for k in ['季节','淡季','旺季','夏天','冬天']):
            self._seasonal()
        elif any(k in t for k in ['先行','放苗','苗','饲料','库存','存塘']):
            self._leading()
        elif any(k in t for k in ['因素','影响','分析']):
            self._factors()
        elif any(k in t for k in ['图','chart','可视化','plot']):
            self._chart()
        elif any(k in t for k in ['建议','策略','怎么','如何','出塘']):
            self._advice()
        else:
            print(f"  未识别：'{text}'  → 输入 'help' 查看命令")


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    agent = TilapiaPriceAgent(data_path=data_path)
    agent.run()
