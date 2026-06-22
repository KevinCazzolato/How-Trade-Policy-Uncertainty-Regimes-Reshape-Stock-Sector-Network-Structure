# ============================================================
# PHASE 1 — Data Acquisition & Preprocessing
# DAI Mission: Trade Policy Uncertainty & Sector Networks
# ============================================================
#
# DATA SOURCES:
#   - Sector ETFs   : yfinance (auto_adjust=True)
#   - Macro controls: FRED via pandas_datareader
#   - TPU Index     : All_Daily_TPU_Data.csv (Baker, Bloom & Davis)
#
# FIXES APPLIED:
#   1. MultiIndex flattening after yf.download
#   2. dropna(subset=["vix"]) only — realized_vol NaNs documented
#   3. FEDFUNDS replaced with DFF (daily effective fed funds rate)
#   4. ADF tests extended to macro & TPU series
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


class DataPipeline:

    TICKERS = [
        "XLK",  # Technology
        "XLI",  # Industrials
        "XLY",  # Consumer Discretionary
        "XLF",  # Financials
        "XLV",  # Health Care
        "XLE",  # Energy
        "XLB",  # Materials
        "XLRE",  # Real Estate
        "XLU",  # Utilities
        "XLP",  # Consumer Staples
        "XLC",  # Communication Services
    ]

    SECTOR_NAMES = {
        "XLK": "Technology",
        "XLI": "Industrials",
        "XLY": "Cons. Discr.",
        "XLF": "Financials",
        "XLV": "Health Care",
        "XLE": "Energy",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLP": "Cons. Staples",
        "XLC": "Communication",
    }

    def __init__(self, start_date="2018-01-01", end_date="2026-06-18"):
        self.start = start_date
        self.end = end_date

        try:
            # Funziona quando il file viene eseguito come script (src/data_acquisition.py)
            self.base_dir = Path(__file__).resolve().parents[1]
        except NameError:
            # Fallback per notebook Jupyter, dove __file__ non esiste:
            # assume che Jupyter sia lanciato dalla root della repo.
            self.base_dir = Path.cwd()

        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════
    # 1. SECTOR ETF PRICES & VOLUME — yfinance
    #    FIX 1: yf.download returns a MultiIndex when downloading
    #    multiple tickers. We flatten columns immediately after
    #    download to guarantee a clean list of ticker strings.
    # ════════════════════════════════════════════════════════
    def fetch_etf_data(self):
        print("📥 Download ETF prices from Yahoo Finance...")
        raw_yf = yf.download(
            self.TICKERS,
            start=self.start,
            end=self.end,
            auto_adjust=True,  # adjusts for splits & dividends
            progress=False,
        )

        prices_wide = raw_yf["Close"][self.TICKERS].sort_index()
        prices_wide.columns = self.TICKERS  # force flat string columns

        volume_wide = raw_yf["Volume"][self.TICKERS].sort_index()
        volume_wide.columns = self.TICKERS

        print(f"✓ yfinance loaded — {len(prices_wide)} trading days")
        print(
            f"  Date range  : {prices_wide.index[0].date()} → {prices_wide.index[-1].date()}"
        )
        print(f"  Price matrix: {prices_wide.shape}")
        print(prices_wide.tail(3).round(2))

        # ──────────────────────────────────────────────────
        # 2. DERIVED SERIES
        # ──────────────────────────────────────────────────
        # Log-returns (stationary — HMM, Granger networks, supervised learning)
        log_returns = np.log(prices_wide / prices_wide.shift(1)).dropna()

        # Log-prices (non-stationary levels — cointegration / VECM)
        log_prices = np.log(prices_wide)

        # Realised volatility: rolling 21-day, annualised.
        # NOTE: first 21 rows will be NaN by construction. The master
        # dataset will therefore start ~21 trading days later.
        # This is documented and expected — see missing value report.
        realized_vol = log_returns.rolling(21).std() * np.sqrt(252)

        print(f"\nLog-returns : {log_returns.shape}")
        print(log_returns.describe().round(4))

        return log_returns, log_prices, realized_vol, volume_wide

    # ════════════════════════════════════════════════════════
    # 3. MACRO CONTROLS FROM FRED
    #    FIX 3: replaced FEDFUNDS (monthly) with DFF (daily
    #    effective fed funds rate). DFF is natively daily — no
    #    frequency mismatch, no artificial publication-lag on
    #    monthly data.
    # ════════════════════════════════════════════════════════
    def fetch_macro_controls(self):
        print("\n📥 Download macro controls from FRED...")
        vix = web.DataReader("VIXCLS", "fred", self.start, self.end).rename(
            columns={"VIXCLS": "vix"}
        )
        spread = web.DataReader("T10Y2Y", "fred", self.start, self.end).rename(
            columns={"T10Y2Y": "spread_10y2y"}
        )
        dff = web.DataReader("DFF", "fred", self.start, self.end).rename(
            columns={"DFF": "dff"}
        )

        macro_raw = vix.join([spread, dff], how="outer").ffill()
        print("✓ Macro controls loaded  [VIX | spread_10y2y | DFF]")
        print(macro_raw.tail(3).round(3))
        return macro_raw

    # ════════════════════════════════════════════════════════
    # 4. TPU INDEX  (Baker, Bloom & Davis — Daily TPU)
    #    File   : All_Daily_TPU_Data.csv  (cartella self.data_dir)
    #    Columns: day, month, year, daily_tpu_index
    #    Zeros → forward-filled (vedi motivazione nel codice)
    # ════════════════════════════════════════════════════════
    def load_tpu_index(self, filename="All_Daily_TPU_Data.csv"):
        tpu_path = self.data_dir / filename
        if not tpu_path.exists():
            raise FileNotFoundError(
                f"❌ File {filename} non trovato in {self.data_dir}. "
                "Assicurati di aver inserito il dataset nella cartella corretta della repository."
            )

        print(f"\n📖 Loading TPU Index from {tpu_path.name}...")
        tpu_raw = pd.read_csv(tpu_path)
        tpu_raw["date"] = pd.to_datetime(tpu_raw[["year", "month", "day"]])
        tpu_raw = (
            tpu_raw.set_index("date")[["daily_tpu_index"]]
            .rename(columns={"daily_tpu_index": "tpu"})
            .sort_index()
        )

        tpu_raw = tpu_raw.loc[self.start : self.end]

        # Motivation for zero diagnostics:
        # We perform a lag/lead inspection around zero values (t-1, t, t+1) to assess
        # whether zeros represent genuine economic observations or data artifacts.
        # Given the structure of the TPU index (news-based frequency measure), it is
        # highly implausible for true zero uncertainty to occur; thus, observed zeros
        # are interpreted as missing/absent observations rather than meaningful values,
        # therefore they are replaced with NaN and forward-filled.
        tpu_raw["tpu"] = tpu_raw["tpu"].replace(0, float("nan")).ffill()

        print(f"✓ TPU index loaded — {len(tpu_raw)} calendar days")
        print(f"  Range : {tpu_raw.index[0].date()} → {tpu_raw.index[-1].date()}")
        print(tpu_raw.describe().round(2))
        return tpu_raw

    # ════════════════════════════════════════════════════════
    # 5. ALIGN ALL SERIES ON TRADING DAYS + PUBLICATION-LAG
    #
    #    All macro / TPU variables are lagged by 1 trading day:
    #    information available on day t is used as a feature on t+1.
    #
    #    FIX 2: dropna is applied ONLY on ["vix"] — this preserves
    #    all trading days including the realized_vol warmup period.
    #    The _rvol columns will show NaN for the first ~21 rows;
    #    this is handled downstream (e.g. imputed or excluded per model).
    # ════════════════════════════════════════════════════════
    def build_master_dataset(self, log_returns, macro_raw, tpu_raw, realized_vol):
        print("\n🔗 Aligning series and applying 1-day publication lag...")
        trading_days = log_returns.index
        tpu_daily = tpu_raw.reindex(macro_raw.index, method="ffill")

        macro_lagged = macro_raw.reindex(trading_days).ffill().shift(1)
        tpu_lagged = tpu_daily.reindex(trading_days).ffill().shift(1)
        rvol_lagged = realized_vol.shift(1)

        master = (
            log_returns.join(macro_lagged)
            .join(tpu_lagged)
            .join(rvol_lagged, rsuffix="_rvol")
            .dropna(subset=["vix"])  # only drop rows where VIX is missing
        )
        master.index.name = "date"

        print(f"\nMaster dataset: {master.shape[0]} rows × {master.shape[1]} columns")
        print(
            f"  Effective start : {master.index[0].date()}  "
            f"(~21 trading days after {log_returns.index[0].date()} due to rvol warmup)"
        )
        print(master.head(3).round(4))

        return master, tpu_daily

    # ════════════════════════════════════════════════════════
    # 6. STATIONARITY CHECKS (ADF TESTS)
    #    FIX 4: extended to macro and TPU series in addition to ETFs.
    #    For non-stationary macro series, first differences are tested.
    # ════════════════════════════════════════════════════════
    @staticmethod
    def _adf_test(series, name):
        s = series.dropna()
        stat, pval, _, nobs, _, _ = adfuller(s, autolag="AIC")
        result = "✓ stationary" if pval < 0.05 else "✗ unit root"
        return stat, pval, nobs, result

    @classmethod
    def run_adf_tests(cls, series_dict, label):
        print(f"\n── ADF tests: {label} ──")
        print(
            f"{'Series':<18}  {'ADF stat':>10}  {'p-value':>10}  {'N':>6}  {'Result':>14}"
        )
        print("-" * 66)
        for name, s in series_dict.items():
            stat, pval, nobs, result = cls._adf_test(s, name)
            print(f"{name:<18}  {stat:>10.3f}  {pval:>10.4f}  {nobs:>6}  {result:>14}")

    def run_full_stationarity_report(
        self, log_returns, log_prices, macro_raw, tpu_daily
    ):
        # ETF log-returns (expect all stationary)
        self.run_adf_tests(
            {
                col: log_returns[col]
                for col in self.TICKERS
                if col in log_returns.columns
            },
            "ETF Log-returns  (expect: stationary)",
        )

        # ETF log-prices (expect all unit root)
        self.run_adf_tests(
            {col: log_prices[col] for col in self.TICKERS if col in log_prices.columns},
            "ETF Log-prices  (expect: unit root → levels for VECM)",
        )

        # Macro & TPU series (mixed — check each)
        macro_tpu = {
            "vix": macro_raw["vix"],
            "spread_10y2y": macro_raw["spread_10y2y"],
            "dff": macro_raw["dff"],
            "tpu": tpu_daily["tpu"],
        }
        self.run_adf_tests(macro_tpu, "Macro & TPU series  (check carefully)")

        # First differences for any non-stationary macro series
        macro_tpu_diff = {f"Δ{k}": v.diff() for k, v in macro_tpu.items()}
        self.run_adf_tests(macro_tpu_diff, "First differences")

    # ════════════════════════════════════════════════════════
    # 7. MISSING VALUE REPORT
    # ════════════════════════════════════════════════════════
    @staticmethod
    def missing_value_report(master):
        print("\n── Missing value summary (% of rows) ──")
        miss = master.isnull().mean().mul(100).round(2)
        nonempty = miss[miss > 0]
        if nonempty.empty:
            print("No missing values ✓")
            return

        print(nonempty.to_string())
        rvol_miss = nonempty[nonempty.index.str.contains("rvol")]
        if not rvol_miss.empty:
            print(
                f"\n  NOTE: _rvol NaNs ({rvol_miss.iloc[0]:.1f}%) are the 21-day warmup period."
            )
            print(f"  Master dataset starts: {master.index[0].date()}")
            rvol_cols = [c for c in master.columns if c.endswith("_rvol")]
            if rvol_cols:
                warmup_end = master[master[rvol_cols[0]].notna()].index[0]
                print(f"  Full rvol available from: {warmup_end.date()}")
                print(
                    "  → For HMM/network models use full master; "
                    "for rvol features start from warmup_end."
                )

    # ════════════════════════════════════════════════════════
    # 8. SAVE OUTPUTS
    # ════════════════════════════════════════════════════════
    def save_outputs(
        self, log_returns, log_prices, realized_vol, master, volume_wide=None
    ):
        log_returns.to_csv(self.data_dir / "log_returns.csv")
        log_prices.to_csv(self.data_dir / "log_prices.csv")
        realized_vol.to_csv(self.data_dir / "realized_vol.csv")
        master.to_csv(self.data_dir / "master_dataset.csv")
        if volume_wide is not None:
            volume_wide.to_csv(self.data_dir / "volume.csv")

        print(f"\n💾 Files saved to: {self.data_dir}")
        print("   log_returns.csv    — daily log-returns (stationary)")
        print("   log_prices.csv     — daily log-prices  (levels for VECM)")
        print("   realized_vol.csv   — 21-day rolling annualised volatility")
        print("   master_dataset.csv — aligned features with 1-day lag on macro/TPU")
        if volume_wide is not None:
            print("   volume.csv         — daily traded volume per ETF")

        print(f"\nTickers     : {list(log_returns.columns)}")
        print(
            f"Date range  : {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        )
        print(f"Trading days: {len(log_returns)}")

        tpu_missing = master["tpu"].isna().mean() * 100
        if tpu_missing > 5:
            print(
                f"\n⚠️  WARNING: TPU has {tpu_missing:.1f}% missing — check the TPU csv path."
            )
        else:
            print(f"✓ TPU coverage: {100 - tpu_missing:.1f}% of trading days")

    # ════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ════════════════════════════════════════════════════════
    def run_pipeline(self):
        # 1-2. ETF prices, volume & derived series
        log_returns, log_prices, realized_vol, volume_wide = self.fetch_etf_data()

        # 3. Macro controls
        macro_raw = self.fetch_macro_controls()

        # 4. TPU index
        tpu_raw = self.load_tpu_index()

        # 5. Align + lag
        master, tpu_daily = self.build_master_dataset(
            log_returns, macro_raw, tpu_raw, realized_vol
        )

        # 6. Stationarity checks
        self.run_full_stationarity_report(log_returns, log_prices, macro_raw, tpu_daily)

        # 7. Missing value report
        self.missing_value_report(master)

        # 8. Save outputs
        self.save_outputs(log_returns, log_prices, realized_vol, master, volume_wide)

        return log_returns, log_prices, macro_raw, tpu_raw, master


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_pipeline()
