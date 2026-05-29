# ==============================================================================
# 頂尖對沖基金級：多維度動能與跨資產避險網格搜索引擎 (Colab 極短線高頻版)
# ==============================================================================
import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from google.colab import drive
import warnings
import sys

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ==============================================================================
# 0. 系統參數與環境設定
# ==============================================================================
drive.mount('/content/drive')

# ✅ 您的正確路徑
BASE_DIR = '/content/drive/MyDrive/AIagent/backtest_database'

# 選擇市場切換器
MARKET = 'US'

if MARKET == 'US':
    STOCK_DIR = os.path.join(BASE_DIR, 'us_stocks')
    ETF_DIR = os.path.join(BASE_DIR, 'us_etfs')
    MACRO_DIR = os.path.join(BASE_DIR, 'macro_indices')
    BENCHMARK_TICKER = 'SPY'
    HEDGE_TICKER = 'TLT'
    INIT_CAPITAL = 10000.0
else:
    STOCK_DIR = os.path.join(BASE_DIR, 'taiwan_stocks')
    ETF_DIR = os.path.join(BASE_DIR, 'taiwan_etfs')
    MACRO_DIR = os.path.join(BASE_DIR, 'macro_indices')
    BENCHMARK_TICKER = '0050'
    HEDGE_TICKER = '00679B'
    INIT_CAPITAL = 300000.0

# 🔥 更新：依照您的指示，全面改為極短線與高避險參數
PARAM_GRID = {
    'top_n': [3, 5, 8, 10],
    'mom_lookback_days': [1, 3, 5, 10, 21],
    'rebalance_freq': ['1D', '3D', '1W', '2W', '1M'],
    'ma_filter': [0, 5, 10, 20],
    'mom_filter': ['None', '1D', '3D', '5D', '10D'],
    'hedge_ratio': [0.0, 0.3, 0.5, 0.8]
}

# 熊市評估區間設定
PERIODS = {
    '2000_Bear': ('2000-01-01', '2004-12-31'),
    '2008_Bear': ('2008-01-01', '2012-12-31'),
    '2020_Bear': ('2020-01-01', '2021-12-31'),
    '2022_Bear': ('2022-01-01', '2023-12-31')
}

# ==============================================================================
# 1. 本地資料批量讀取模組
# ==============================================================================
def load_data(folder_path):
    if not os.path.exists(folder_path): return pd.DataFrame()
    files = glob.glob(os.path.join(folder_path, '*.csv'))
    df_list = []
    for f in files:
        ticker = os.path.basename(f).split('_')[0].replace('.csv', '')
        try:
            df = pd.read_csv(f, parse_dates=['Date'])
            if 'Adj Close' not in df.columns: df['Adj Close'] = df['Close']
            df = df[['Date', 'Adj Close']].rename(columns={'Adj Close': ticker})
            df.set_index('Date', inplace=True)
            df_list.append(df)
        except:
            pass
    if not df_list: return pd.DataFrame()
    master_df = pd.concat(df_list, axis=1).sort_index().ffill()
    return master_df

print("⏳ 正在掃描並載入本機 CSV 資料庫...")
price_data = load_data(STOCK_DIR)
macro_data = load_data(MACRO_DIR)
etf_data = load_data(ETF_DIR)

if price_data.empty: sys.exit(f"❌ 錯誤：在 {STOCK_DIR} 找不到股票資料。")

def get_or_download_ticker(ticker, dataframes):
    for df in dataframes:
        if not df.empty and ticker in df.columns: return df[ticker]
    print(f"⚠️ 啟動 Wildcard 補齊 {ticker}...")
    try:
        tmp = yf.download(ticker, start="1996-01-01", progress=False, auto_adjust=False)
        if 'Adj Close' in tmp:
            s = tmp['Adj Close'].squeeze()
            s.index = s.index.tz_localize(None)
            return s
    except: pass
    return None

benchmark_price = get_or_download_ticker(BENCHMARK_TICKER, [macro_data, etf_data, price_data])
hedge_price = get_or_download_ticker(HEDGE_TICKER, [macro_data, etf_data, price_data])

if benchmark_price is None or hedge_price is None: sys.exit("❌ 缺乏大盤或避險資料。")

valid_dates = price_data.index.intersection(benchmark_price.index).intersection(hedge_price.index)
price_data = price_data.loc[valid_dates]
benchmark_price = benchmark_price.loc[valid_dates]
hedge_price = hedge_price.loc[valid_dates]

daily_returns = price_data.pct_change().fillna(0)
hedge_returns = hedge_price.pct_change().fillna(0)

# ==============================================================================
# 2. 核心回測引擎 (向量化矩陣加速)
# ==============================================================================
def calculate_metrics(nav_series):
    if len(nav_series) < 2: return 0, 0, 0, 0
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    mdd = (nav_series / nav_series.cummax() - 1.0).min()

    rf = 0.03
    rets = nav_series.pct_change().dropna()
    downside_rets = rets[rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (rets.mean() * 252 - rf) / downside_std if downside_std > 0 else 0

    std = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / std if std > 0 else 0

    return cagr, mdd, sortino, sharpe

def run_strategy(top_n, mom_days, reb_freq, ma_filter, mom_filter, hedge_ratio):
    momentum_scores = price_data.pct_change(periods=mom_days)
    safe_signal = pd.Series(True, index=benchmark_price.index)

    # 均線濾網
    if ma_filter > 0:
        ma_series = benchmark_price.rolling(window=ma_filter).mean()
        safe_signal = safe_signal & (benchmark_price >= ma_series)

    # 🔥 更新：支援極短線大盤動能邏輯 (1D, 3D, 5D, 10D)
    if mom_filter != 'None':
        if mom_filter == '1D':
            safe_signal = safe_signal & (benchmark_price.pct_change(1) > 0)
        elif mom_filter == '3D':
            safe_signal = safe_signal & (benchmark_price.pct_change(3) > 0)
        elif mom_filter == '5D':
            safe_signal = safe_signal & (benchmark_price.pct_change(5) > 0)
        elif mom_filter == '10D':
            safe_signal = safe_signal & (benchmark_price.pct_change(10) > 0)

    # 生成換股日
    if reb_freq == '1D':
        reb_dates = price_data.index
    elif reb_freq == '3D':
        reb_dates = price_data.index[::3]
    elif reb_freq == '1W':
        reb_dates = price_data.index[::5]
    elif reb_freq == '2W':
        reb_dates = price_data.index[::10]
    elif reb_freq == '1M':
        ym = price_data.index.year.astype(str) + price_data.index.month.astype(str)
        reb_dates = pd.DatetimeIndex(pd.Series(price_data.index, index=price_data.index).groupby(ym).last())

    reb_mom = momentum_scores.loc[reb_dates]
    reb_safe = safe_signal.loc[reb_dates]

    ranks = reb_mom.rank(axis=1, ascending=False, method='first')
    buy_mask = (ranks <= top_n).values & reb_safe.values[:, None]

    stock_w = pd.DataFrame(buy_mask, index=reb_dates, columns=momentum_scores.columns).astype(float) / top_n
    hedge_w = pd.Series((~reb_safe).astype(float) * hedge_ratio, index=reb_dates)

    full_stock_w = stock_w.reindex(price_data.index).ffill().shift(1).fillna(0)
    full_hedge_w = hedge_w.reindex(price_data.index).ffill().shift(1).fillna(0)

    strat_daily_returns = (daily_returns * full_stock_w).sum(axis=1) + (hedge_returns * full_hedge_w)

    strat_nav = INIT_CAPITAL * (1 + strat_daily_returns).cumprod()
    return strat_nav

# ==============================================================================
# 3. 執行網格搜索與 4:6 綜合評分
# ==============================================================================
combinations = list(itertools.product(
    PARAM_GRID['top_n'], PARAM_GRID['mom_lookback_days'], PARAM_GRID['rebalance_freq'],
    PARAM_GRID['ma_filter'], PARAM_GRID['mom_filter'], PARAM_GRID['hedge_ratio']
))

total_runs = len(combinations)
print(f"\n🚀 開始執行極短線網格搜索，總計 {total_runs} 種組合 (向量化引擎啟動)...")

results = []
for idx, (n, md, rf_freq, maf, momf, hr) in enumerate(combinations):
    if idx % 500 == 0:
        print(f"   ► 進度: {idx}/{total_runs} ({(idx/total_runs)*100:.1f}%)")

    nav = run_strategy(n, md, rf_freq, maf, momf, hr)
    cagr_all, mdd_all, sortino_all, sharpe_all = calculate_metrics(nav)

    sortino_bears = []
    for period_name, (start, end) in PERIODS.items():
        try:
            period_nav = nav.loc[start:end]
            _, _, s_bear, _ = calculate_metrics(period_nav)
            sortino_bears.append(s_bear)
        except:
            sortino_bears.append(0)

    weighted_sortino = (0.25 * sortino_bears[0]) + (0.25 * sortino_bears[1]) + (0.25 * sortino_bears[2]) + (0.25 * sortino_bears[3])

    results.append({
        'Top_N': n, 'Mom_Days': md, 'Rebalance_Freq': rf_freq,
        'MA_Filter': maf, 'Mom_Filter': momf, 'Hedge_Ratio': hr,
        'CAGR': cagr_all, 'MDD': mdd_all, 'Sortino': sortino_all, 'Sharpe': sharpe_all,
        'Sortino_2000': sortino_bears[0], 'Sortino_2008': sortino_bears[1],
        'Sortino_2020': sortino_bears[2], 'Sortino_2022': sortino_bears[3],
        'Weighted_Sortino': weighted_sortino, '_NAV_Series': nav
    })

df_res = pd.DataFrame(results)

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

df_res['Norm_CAGR'] = min_max_scale(df_res['CAGR'])
df_res['Norm_WSortino'] = min_max_scale(df_res['Weighted_Sortino'])
df_res['Final_Score'] = (0.4 * df_res['Norm_CAGR']) + (0.6 * df_res['Norm_WSortino'])

# 排序出最強 Top 10
df_res = df_res.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
top_10 = df_res.head(10)

# ==============================================================================
# 4. 視覺化與匯出
# ==============================================================================
print(f"\n📊 正在生成並儲存 Top 10 策略的四宮格壓力測試走勢圖...")

for i in range(len(top_10)):
    strategy = top_10.iloc[i]
    strat_nav_full = strategy['_NAV_Series']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"[{MARKET}] Rank {i+1} (Score: {strategy['Final_Score']:.3f})\n"
                 f"Top N: {strategy['Top_N']} | Mom Days: {strategy['Mom_Days']} | Freq: {strategy['Rebalance_Freq']} | MA: {strategy['MA_Filter']} | Hedge: {strategy['Hedge_Ratio']*100}%",
                 fontsize=16, y=0.98)

    axes = axes.flatten()
    for j, (period_name, (start, end)) in enumerate(PERIODS.items()):
        ax = axes[j]
        try:
            strat_slice = strat_nav_full.loc[start:end]
            bench_slice = benchmark_price.loc[start:end]

            if len(strat_slice) > 0 and len(bench_slice) > 0:
                rebased_strat = (strat_slice / strat_slice.iloc[0]) * INIT_CAPITAL
                rebased_bench = (bench_slice / bench_slice.iloc[0]) * INIT_CAPITAL

                ax.plot(rebased_strat.index, rebased_strat, label="Strategy", color='#00ffff', linewidth=2)
                ax.plot(rebased_bench.index, rebased_bench, label=f"Buy & Hold {BENCHMARK_TICKER}", color='white', linestyle='--', alpha=0.6)

                c_years = (rebased_strat.index[-1] - rebased_strat.index[0]).days / 365.25
                c_cagr = (rebased_strat.iloc[-1] / rebased_strat.iloc[0]) ** (1 / c_years) - 1 if c_years > 0 else 0
                c_mdd = (rebased_strat / rebased_strat.cummax() - 1.0).min()

                ax.set_title(f"{period_name}\nCAGR: {c_cagr*100:.1f}% | MDD: {c_mdd*100:.1f}%", fontsize=14)
                ax.set_ylabel("Portfolio Value", fontsize=12)
                ax.set_yscale('log')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.2)
        except:
            pass

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    img_name = f'/content/drive/MyDrive/AIagent/top_{i+1}_equity_curve_{MARKET}.png'
    plt.savefig(img_name)
    if i == 0: plt.show()
    plt.close()

export_df = top_10.drop(columns=['_NAV_Series', 'Norm_CAGR', 'Norm_WSortino'])
csv_name = f'/content/drive/MyDrive/AIagent/{MARKET}_strategy_leaderboard.csv'
export_df.to_csv(csv_name, index=False)

print(f"\n✅ {total_runs} 種組合計算完畢！已成功匯出至：{os.path.dirname(csv_name)}")
best_strategy = top_10.iloc[0]
print(f"🥇 【第 1 名參數配置】")
print(f" - 動能持股檔數: {best_strategy['Top_N']} 檔")
print(f" - 動能計算週期: {best_strategy['Mom_Days']} 天")
print(f" - 換股調倉頻率: {best_strategy['Rebalance_Freq']}")
print(f" - 大盤均線濾網: {best_strategy['MA_Filter']} MA")
print(f" - 大盤動能濾網: {best_strategy['Mom_Filter']}")
print(f" - 觸發避險比例: {best_strategy['Hedge_Ratio'] * 100}%")
print(f"🥇 【全區間整體績效】")
print(f" - 年化報酬率 (CAGR): {best_strategy['CAGR'] * 100:.2f}%")
print(f" - 最大回撤率 (MDD): {best_strategy['MDD'] * 100:.2f}%")
