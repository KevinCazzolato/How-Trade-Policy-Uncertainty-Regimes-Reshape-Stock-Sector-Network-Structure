import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


class DataPipeline:

    def __init__(self, start_date="2018-01-01", end_date="2026-06-18"):
        self.start = start_date
        self.end = end_date

        # Gestione Path Relativi Universali (Trova la radice del progetto)
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tickers = [
            "XLK",
            "XLI",
            "XLY",
            "XLF",
            "XLV",
            "XLE",
            "XLB",
            "XLRE",
            "XLU",
            "XLP",
            "XLC",
        ]

    def fetch_etf_data(self):
        print("📥 Download ETF prices from Yahoo Finance...")
        raw_yf = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        prices = raw_yf["Close"][self.tickers].sort_index()
        prices.columns = self.tickers

        log_returns = np.log(prices / prices.shift(1)).dropna()
        log_prices = np.log(prices)
        realized_vol = log_returns.rolling(21).std() * np.sqrt(252)

        return log_returns, log_prices, realized_vol

    def fetch_macro_controls(self):
        print("📥 Download macro controls from FRED...")
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
        return macro_raw

    def load_tpu_index(self, filename="All_Daily_TPU_Data.csv"):
        tpu_path = self.data_dir / filename
        if not tpu_path.exists():
            raise FileNotFoundError(
                f"❌ File {filename} non trovato in {self.data_dir}. "
                "Assicurati di aver inserito il dataset nella cartella corretta della repository."
            )

        print(f"📖 Loading TPU Index from {tpu_path.name}...")
        tpu_raw = pd.read_csv(tpu_path)
        tpu_raw["date"] = pd.to_datetime(tpu_raw[["year", "month", "day"]])
        tpu_raw = (
            tpu_raw.set_index("date")[["daily_tpu_index"]]
            .rename(columns={"daily_tpu_index": "tpu"})
            .sort_index()
        )

        tpu_raw = tpu_raw.loc[self.start : self.end]
        # Gestione Zeros Economici (BBD Index diagnostics)
        tpu_raw["tpu"] = tpu_raw["tpu"].replace(0, float("nan")).ffill()
        return tpu_raw

    def build_master_dataset(self, log_returns, macro_raw, tpu_raw, realized_vol):
        print("🔗 Aligning series and applying 1-day publication lag...")
        trading_days = log_returns.index
        tpu_daily = tpu_raw.reindex(macro_raw.index, method="ffill")

        macro_lagged = macro_raw.reindex(trading_days).ffill().shift(1)
        tpu_lagged = tpu_daily.reindex(trading_days).ffill().shift(1)
        rvol_lagged = realized_vol.shift(1)

        master = (
            log_returns.join(macro_lagged)
            .join(tpu_lagged)
            .join(rvol_lagged, rsuffix="_rvol")
            .dropna(subset=["vix"])
        )
        master.index.name = "date"
        return master

    @staticmethod
    def run_adf_tests(series_dict, label):
        print(f"\n📊 ADF tests: {label}")
        print(f"{'Series':<18}  {'ADF stat':>10}  {'p-value':>10}  {'Result'}")
        print("-" * 60)
        for name, s in series_dict.items():
            stat, pval, _, _, _, _ = adfuller(s.dropna(), autolag="AIC")
            res = "✓ stationary" if pval < 0.05 else "✗ unit root"
            print(f"{name:<18}  {stat:>10.3f}  {pval:>10.4f}  {res}")

    def run_pipeline(self):
        # 1. Get Data
        log_returns, log_prices, realized_vol = self.fetch_etf_data()
        macro_raw = self.fetch_macro_controls()
        tpu_raw = self.load_tpu_index()

        # 2. Build Master
        master = self.build_master_dataset(
            log_returns, macro_raw, tpu_raw, realized_vol
        )

        # 3. Save Outputs (in modo relativo alla repo)
        log_returns.to_csv(self.data_dir / "log_returns.csv")
        log_prices.to_csv(self.data_dir / "log_prices.csv")
        realized_vol.to_csv(self.data_dir / "realized_vol.csv")
        master.to_csv(self.data_dir / "master_dataset.csv")
        print(f"💾 All datasets saved into: {self.data_dir}")

        return log_returns, log_prices, macro_raw, tpu_raw, master


if __name__ == "__main__":
    # Test esecuzione diretta dello script
    pipeline = DataPipeline()
    pipeline.run_pipeline()
