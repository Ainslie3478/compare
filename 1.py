# ==============================================================================
# 頂尖對沖基金級：多維度動能與跨資產避險網格搜索引擎 (Colab 終極無錯版)
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

# 您指定的網格搜索參數空間
PARAM_GRID = {
    'top_n': [3, 5, 8, 10],
    'mom_lookback_days': [5, 10, 15, 21, 63], # 1周, 2周, 3周, 1月, 3月
    'ma_filter': [0, 20, 50, 200],            # 0 代表無濾網
    'mom_filter': ['None', '1M', '1M+3M', '1M+3M+6M'],
    'hedge_ratio': [0.0, 0.2, 0.4, 0.5]
}

# 熊市評估區間設定 (改為四個完整區間)
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

if price_data.empty:
    sys.exit(f"❌ 錯誤：在 {STOCK_DIR} 找不到股票資料。")

# --- 啟動 Wildcard 條款 (自動補齊大盤與避險標的) ---
def get_or_download_ticker(ticker, dataframes):
    # 先從本地各資料夾尋找
    for df in dataframes:
        if not df.empty and ticker in df.columns:
            print(f"✅ 在本地成功找到 {ticker} 的資料。")
            return df[ticker]
    # 若本地沒有，觸發 Wildcard 下載
    print(f"⚠️ 本地找不到 {ticker}，啟動 Wildcard 透過 yfinance 補齊下載...")
    try:
        tmp = yf.download(ticker, start="1996-01-01", progress=False, auto_adjust=False)
        if 'Adj Close' in tmp:
            s = tmp['Adj Close'].squeeze()
            s.index = s.index.tz_localize(None) # 移除時區確保能對齊
            return s
    except: pass
    return None

benchmark_price = get_or_download_ticker(BENCHMARK_TICKER, [macro_data, etf_data, price_data])
hedge_price = get_or_download_ticker(HEDGE_TICKER, [macro_data, etf_data, price_data])

if benchmark_price is None: sys.exit("❌ 無法取得大盤資料，程式終止。")
if hedge_price is None: sys.exit("❌ 無法取得避險資料，程式終止。")

# 統一時間軸 (只保留大家都有交易的日子)
valid_dates = price_data.index.intersection(benchmark_price.index).intersection(hedge_price.index)
price_data = price_data.loc[valid_dates]
benchmark_price = benchmark_price.loc[valid_dates]
hedge_price = hedge_price.loc[valid_dates]

daily_returns = price_data.pct_change().fillna(0)
hedge_returns = hedge_price.pct_change().fillna(0)

# 🔥 核心修正：取得每個月「最後一個有開盤的實際交易日」，消滅 KeyError
month_periods = price_data.index.to_period('M')
rebalance_dates = pd.Series(price_data.index, index=price_data.index).groupby(month_periods).last()
rebalance_dates = pd.DatetimeIndex(rebalance_dates)

# ==============================================================================
# 2. 核心回測引擎 (向量化加速)
# ==============================================================================
def calculate_metrics(nav_series):
    if len(nav_series) < 2: return 0, 0, 0, 0
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    roll_max = nav_series.cummax()
    drawdown = nav_series / roll_max - 1.0
    mdd = drawdown.min()

    rf = 0.03
    rets = nav_series.pct_change().dropna()
    downside_rets = rets[rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (rets.mean() * 252 - rf) / downside_std if downside_std > 0 else 0

    std = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / std if std > 0 else 0

    return cagr, mdd, sortino, sharpe

def run_strategy(top_n, mom_days, ma_filter, mom_filter, hedge_ratio):
    momentum_scores = price_data.pct_change(periods=mom_days)
    safe_signal = pd.Series(1, index=benchmark_price.index)

    # 均線濾網
    if ma_filter > 0:
        ma_series = benchmark_price.rolling(window=ma_filter).mean()
        safe_signal = safe_signal & (benchmark_price >= ma_series)

    # 動能濾網
    if mom_filter != 'None':
        mom_1m = benchmark_price.pct_change(21)
        if mom_filter == '1M':
            safe_signal = safe_signal & (mom_1m > 0)
        elif mom_filter == '1M+3M':
            mom_3m = benchmark_price.pct_change(63)
            safe_signal = safe_signal & (((mom_1m + mom_3m) / 2) > 0)
        elif mom_filter == '1M+3M+6M':
            mom_3m = benchmark_price.pct_change(63)
            mom_6m = benchmark_price.pct_change(126)
            safe_signal = safe_signal & (((mom_1m + mom_3m + mom_6m) / 3) > 0)

    portfolio_nav = pd.Series(index=price_data.index, dtype=float)
    portfolio_nav.iloc[0] = INIT_CAPITAL
    strat_daily_returns = pd.Series(0.0, index=price_data.index)

    # 調倉迴圈
    for i in range(1, len(rebalance_dates)):
        start_date = rebalance_dates[i-1]
        end_date = rebalance_dates[i]

        # start_date 保證在 index 內，不會再報錯
        is_safe = safe_signal.loc[start_date]

        if is_safe:
            row = momentum_scores.loc[start_date]
            top_stocks = row.nlargest(top_n).index.tolist()
            period_rets = daily_returns.loc[start_date:end_date, top_stocks].mean(axis=1)
        else:
            period_rets = hedge_returns.loc[start_date:end_date] * hedge_ratio

        strat_daily_returns.loc[start_date:end_date] = period_rets

    strat_nav = INIT_CAPITAL * (1 + strat_daily_returns).cumprod()
    return strat_nav

# ==============================================================================
# 3. 執行網格搜索與 4:6 綜合評分
# ==============================================================================
print(f"\n🚀 開始執行多維度網格搜索，尋找聖杯參數...")
combinations = list(itertools.product(
    PARAM_GRID['top_n'], PARAM_GRID['mom_lookback_days'],
    PARAM_GRID['ma_filter'], PARAM_GRID['mom_filter'], PARAM_GRID['hedge_ratio']
))

total_runs = len(combinations)
print(f"🎯 總共有 {total_runs} 種參數組合等待計算。")

results = []
for idx, (n, md, maf, momf, hr) in enumerate(combinations):
    if idx % 100 == 0:
        print(f"   ► 進度: {idx}/{total_runs} ({(idx/total_runs)*100:.1f}%)")

    nav = run_strategy(n, md, maf, momf, hr)
    cagr_all, mdd_all, sortino_all, sharpe_all = calculate_metrics(nav)

    # 計算四次熊市區間 Sortino
    sortino_bears = []
    for period_name, (start, end) in PERIODS.items():
        try:
            period_nav = nav.loc[start:end]
            _, _, s_bear, _ = calculate_metrics(period_nav)
            sortino_bears.append(s_bear)
        except:
            sortino_bears.append(0)

    # 4 個區間平均權重 (各佔 25%)
    weighted_sortino = (0.25 * sortino_bears[0]) + (0.25 * sortino_bears[1]) + (0.25 * sortino_bears[2]) + (0.25 * sortino_bears[3])

    results.append({
        'Top_N': n, 'Mom_Days': md, 'MA_Filter': maf, 'Mom_Filter': momf, 'Hedge_Ratio': hr,
        'CAGR': cagr_all, 'MDD': mdd_all, 'Sortino': sortino_all, 'Sharpe': sharpe_all,
        'Sortino_2000': sortino_bears[0], 'Sortino_2008': sortino_bears[1],
        'Sortino_2020': sortino_bears[2], 'Sortino_2022': sortino_bears[3],
        'Weighted_Sortino': weighted_sortino, '_NAV_Series': nav
    })

df_res = pd.DataFrame(results)

# Min-Max 正規化與 4:6 權重相加 (根據您的指示)
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

df_res['Norm_CAGR'] = min_max_scale(df_res['CAGR'])
df_res['Norm_WSortino'] = min_max_scale(df_res['Weighted_Sortino'])
df_res['Final_Score'] = (0.4 * df_res['Norm_CAGR']) + (0.6 * df_res['Norm_WSortino'])

# 排序出最強 Top 10
df_res = df_res.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
top_10 = df_res.head(10)

# ==============================================================================
# 4. 視覺化與匯出 (四個區間獨立重新投入本金)
# ==============================================================================
print(f"\n📊 正在生成並儲存 Top 10 策略的四宮格壓力測試走勢圖...")

for i in range(len(top_10)):
    strategy = top_10.iloc[i]
    strat_nav_full = strategy['_NAV_Series']

    # 建立 2x2 的四宮格畫布
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"[{MARKET}] Rank {i+1} Strategy Stress Test (Score: {strategy['Final_Score']:.3f})\n"
                 f"Top N: {strategy['Top_N']} | MA: {strategy['MA_Filter']} | Mom: {strategy['Mom_Filter']} | Hedge: {strategy['Hedge_Ratio']*100}%",
                 fontsize=18, y=0.98)

    axes = axes.flatten()

    for j, (period_name, (start, end)) in enumerate(PERIODS.items()):
        ax = axes[j]
        try:
            # 擷取該區間的淨值與大盤價格
            strat_slice = strat_nav_full.loc[start:end]
            bench_slice = benchmark_price.loc[start:end]

            if len(strat_slice) > 0 and len(bench_slice) > 0:
                # ✅ 核心邏輯：將區間的第一天重置為初始本金 (INIT_CAPITAL)
                rebased_strat = (strat_slice / strat_slice.iloc[0]) * INIT_CAPITAL
                rebased_bench = (bench_slice / bench_slice.iloc[0]) * INIT_CAPITAL

                ax.plot(rebased_strat.index, rebased_strat, label="Strategy", color='#00ffff', linewidth=2)
                ax.plot(rebased_bench.index, rebased_bench, label=f"Buy & Hold {BENCHMARK_TICKER}", color='white', linestyle='--', alpha=0.6)

                # 計算該獨立區間的 CAGR 與 MDD
                c_years = (rebased_strat.index[-1] - rebased_strat.index[0]).days / 365.25
                c_cagr = (rebased_strat.iloc[-1] / rebased_strat.iloc[0]) ** (1 / c_years) - 1 if c_years > 0 else 0
                c_mdd = (rebased_strat / rebased_strat.cummax() - 1.0).min()

                ax.set_title(f"{period_name} ({start[:4]}-{end[:4]})\nCAGR: {c_cagr*100:.1f}% | MDD: {c_mdd*100:.1f}%", fontsize=14)
                ax.set_ylabel("Portfolio Value", fontsize=12)
                ax.set_yscale('log')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.2)
            else:
                ax.set_title(f"{period_name} (無交易資料)", fontsize=14)
        except Exception as e:
            ax.set_title(f"{period_name} (資料繪製錯誤)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93]) # 預留上方大標題空間

    # 儲存圖片
    img_name = f'/content/drive/MyDrive/AIagent/top_{i+1}_equity_curve_{MARKET}.png'
    plt.savefig(img_name)

    # 為了避免 Colab 畫面被圖表洗版，我們只在輸出區塊顯示第 1 名的四宮格圖
    if i == 0:
        plt.show()

    plt.close() # 關閉畫布釋放記憶體

# 儲存 CSV 排行榜
export_df = top_10.drop(columns=['_NAV_Series', 'Norm_CAGR', 'Norm_WSortino'])
csv_name = f'/content/drive/MyDrive/AIagent/{MARKET}_strategy_leaderboard.csv'
export_df.to_csv(csv_name, index=False)

print(f"\n✅ 已成功將 Top 10 排行榜與 10 張「四宮格壓力測試圖」匯出至：{os.path.dirname(csv_name)}")
best_strategy = top_10.iloc[0]
print(f"🥇 【第 1 名參數配置】")
print(f" - 動能持股檔數: {best_strategy['Top_N']} 檔")
print(f" - 動能計算週期: {best_strategy['Mom_Days']} 天")
print(f" - 大盤均線濾網: {best_strategy['MA_Filter']} MA")
print(f" - 大盤動能濾網: {best_strategy['Mom_Filter']}")
print(f" - 觸發避險比例: {best_strategy['Hedge_Ratio'] * 100}%")
print(f"🥇 【全區間整體績效數據】")
print(f" - 年化報酬率 (CAGR): {best_strategy['CAGR'] * 100:.2f}%")
print(f" - 最大回撤率 (MDD): {best_strategy['MDD'] * 100:.2f}%")
print(f" - 綜合熊市抗跌分數: {best_strategy['Weighted_Sortino']:.2f}")
