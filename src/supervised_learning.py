import os
import warnings
import itertools

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Keeping the console clean so we can focus on the actual metrics
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Setting seeds for reproducibility so we get the exact same results every time we run this
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# PROJECT CONFIGURATION
# # We keep all file paths and column names here so it's easy to update if
# # anyone in the team changes their output formats in the previous phases.
# ----------------------------------------------------------------------------
CONFIG = {
    "sectors": ["XLK", "XLI", "XLY", "XLF", "XLV", "XLE",
                "XLB", "XLRE", "XLU", "XLP", "XLC"],

    # File paths mapping to the team's outputs
    "returns_file":  ("phase_3", "purged_returns.csv"),   # date + one column per sector
    "vol_file":      ("",        "realized_vol.csv"),      # date + one column per sector
    "regimes_file":  ("",        "regime_labels.csv"),     # index = date, has regime label
    "macro_file":    ("",        "master_dataset.csv"),    # fallback source for vix / tpu

    # Standard column names
    "date_col": "date",
    "regime_col": "regime_label",
    "high_label": "High-TPU",
    "low_labels": ["Low-TPU", "Mid-TPU"],

    # TVTP-HMM regimes live in a separate file from Phase 2.
    # We map them carefully here so they don't overwrite the standard HMM columns during the merge.
    "tvtp_file":        ("phase_2", "tvtp_regime_labels.csv"),
    "tvtp_regime_col":  "regime_label",
    "tvtp_high_label":  "High-TPU",
    "tvtp_prob_col":    "prob_high_tpu",            # We strictly use the filtered prob to avoid look-ahead bias!
    "tvtp_transition_cols": ["p01_entry", "p10_exit"],  # Capturing the probability of shifting regimes

    # continuous regime feature: use the FILTERED high-TPU probability.
    # NEVER use the *_smooth version -- smoothing uses the whole sample (future)
    # and would leak. Set to None if your regime file has no filtered prob column.
    "regime_prob_col": "prob_high_tpu",

    # macro feature columns to use if found (searched in regimes_file then macro_file)
    "macro_cols": ["vix", "tpu"],

    # modelling params
    "horizon": 5,                # Predicting the market direction 5 business days (~1 week) into the future
    "lag": 1,                    # Shifting all predictors by 1 day so the model doesn't cheat by looking at today's data
    "centrality_window": 60,     # We use a 60-day rolling window to build the sector network dynamically
    "corr_threshold": 0.5,       # Only strong correlations (>0.5) become edges in our network
    "n_splits": 5,               # 5-fold cross-validation
    "embargo": 5,                # Crucial: Drops the last 5 days of training data so the test set remains completely unseen
}


class PredictiveAnalysis:
    def __init__(self, data_dir="data", config=CONFIG):
        self.data_dir = data_dir
        self.cfg = config
        self.df = None
        self.X_cols = None
        self._tvtp_table = None

    # ---- small path helper -------------------------------------------------
    def _path(self, file_tuple):
        """Quick helper to build paths whether they are in a subfolder or the main data dir."""
        sub, name = file_tuple
        return os.path.join(self.data_dir, sub, name) if sub else os.path.join(self.data_dir, name)

    # ---- loading + schema verification ------------------------------------
    def _load_wide(self, file_tuple, value_name):
        """Takes a wide dataset (dates x sectors) and melts it down into a long format for easier merging."""
        path = self._path(file_tuple)
        df = pd.read_csv(path)
        dcol = self.cfg["date_col"]
        if dcol not in df.columns:
            # maybe the date is the (unnamed) index
            df = df.rename(columns={df.columns[0]: dcol})
        df[dcol] = pd.to_datetime(df[dcol])

        present = [s for s in self.cfg["sectors"] if s in df.columns]
        missing = [s for s in self.cfg["sectors"] if s not in df.columns]
        if missing:
            print(f"  [WARNING] {os.path.basename(path)} is missing sectors: {missing}")
        long = df.melt(id_vars=[dcol], value_vars=present,
                       var_name="Sector", value_name=value_name)
        return long

    def load_and_prepare_data(self):
        cfg = self.cfg
        dcol = cfg["date_col"]
        print("\n--- 1. LOADING DATA ---")

        # Bring in the purged returns and volatility metrics
        returns = self._load_wide(cfg["returns_file"], "Return")
        vol = self._load_wide(cfg["vol_file"], "realized_vol")

        # Grab the standard HMM regimes (date-indexed)
        regimes = pd.read_csv(self._path(cfg["regimes_file"]), index_col=0, parse_dates=True)
        regimes.index.name = dcol
        regimes = regimes.reset_index()
        if cfg["regime_col"] not in regimes.columns:
            raise KeyError(f"'{cfg['regime_col']}' not found in regime file. "
                           f"Columns are: {list(regimes.columns)}")

        # Macro features (vix, tpu): We check the regimes file first, then fall back to the master dataset.
        macro_found = {}
        for c in cfg["macro_cols"]:
            if c in regimes.columns:
                macro_found[c] = regimes[[dcol, c]]
        still_missing = [c for c in cfg["macro_cols"] if c not in macro_found]
        if still_missing:
            try:
                macro = pd.read_csv(self._path(cfg["macro_file"]))
                if dcol not in macro.columns:
                    macro = macro.rename(columns={macro.columns[0]: dcol})
                macro[dcol] = pd.to_datetime(macro[dcol])
                for c in still_missing:
                    if c in macro.columns:
                        macro_found[c] = macro[[dcol, c]]
            except FileNotFoundError:
                pass

        # Now let's carefully load the TVTP data without causing column name collisions
        tvtp_feats = []
        try:
            tv = pd.read_csv(self._path(cfg["tvtp_file"]))
            if dcol not in tv.columns:
                tv = tv.rename(columns={tv.columns[0]: dcol})
            tv[dcol] = pd.to_datetime(tv[dcol])

            if cfg["tvtp_regime_col"] in tv.columns:
                tv["is_high_regime_tvtp"] = (tv[cfg["tvtp_regime_col"]] == cfg["tvtp_high_label"]).astype(int)
                tvtp_feats.append("is_high_regime_tvtp")

            if cfg["tvtp_prob_col"] in tv.columns:
                tv = tv.rename(columns={cfg["tvtp_prob_col"]: "tvtp_prob_high"})
                tvtp_feats.append("tvtp_prob_high")

            for tc in cfg.get("tvtp_transition_cols", []):
                if tc in tv.columns:
                    tv = tv.rename(columns={tc: f"tvtp_{tc}"})
                    tvtp_feats.append(f"tvtp_{tc}")

            self._tvtp_table = tv[[dcol] + tvtp_feats]
            print(f"  [OK] Successfully integrated Phase 2 TVTP features: {tvtp_feats}")
        except FileNotFoundError:
            self._tvtp_table = None
            print("  [WARNING] Couldn't find the TVTP file. We need both HMM and TVTP-HMM for the final pipeline!")

        print("\n--- 2. ASSEMBLING THE MASTER FEATURE MATRIX ---")
        # Stitch everything together on Date and Sector
        df = returns.merge(vol, on=[dcol, "Sector"], how="left")

        keep_regime_cols = [dcol, cfg["regime_col"]]
        prob_col = cfg.get("regime_prob_col")
        if prob_col and prob_col in regimes.columns:
            keep_regime_cols.append(prob_col)

        df = df.merge(regimes[keep_regime_cols], on=dcol, how="left")

        if self._tvtp_table is not None:
            df = df.merge(self._tvtp_table, on=dcol, how="left")

        for c, table in macro_found.items():
            df = df.merge(table, on=dcol, how="left")

        # binary regime indicator (standard HMM); TVTP indicator already merged above
        df["is_high_regime"] = (df[cfg["regime_col"]] == cfg["high_label"]).astype(int)

        # Dynamic Network Centrality
        print("  Calculating rolling network centrality...")
        print("  (We do this dynamically instead of using static PageRank to guarantee zero future data leakage.)")
        centrality = self._rolling_centrality(returns)
        df = df.merge(centrality, on=[dcol, "Sector"], how="left")

        # Define our target: What happens 5 days from now?
        df = df.sort_values([dcol, "Sector"]).reset_index(drop=True)
        df["future_return"] = df.groupby("Sector")["Return"].shift(-cfg["horizon"])
        df["target"] = (df["future_return"] > 0).astype(int)

        # Apply the lag constraint: Push all predictors back by 1 day.
        # This is our ultimate defense against look-ahead bias.
        candidate_feats = ["Return", "realized_vol", "pagerank", "is_high_regime"] + list(macro_found.keys())

        if self._tvtp_table is not None:
            candidate_feats += [c for c in self._tvtp_table.columns if c != dcol]
        if cfg.get("regime_prob_col") and cfg["regime_prob_col"] in df.columns:
            candidate_feats.append(cfg["regime_prob_col"])

        for feat in candidate_feats:
            if feat in df.columns:
                df[f"{feat}_lag{cfg['lag']}"] = df.groupby("Sector")[feat].shift(cfg["lag"])

        self.df = df.dropna().reset_index(drop=True)
        self.X_cols = [c for c in self.df.columns if c.endswith(f"_lag{cfg['lag']}")]

        print(f"\n  Matrix ready. We have {len(self.X_cols)} lagged features: {self.X_cols}")
        print(
            f"  Valid training rows: {len(self.df)} | Timeline: {self.df[dcol].min().date()} to {self.df[dcol].max().date()}")
        print(f"  Target distribution (Positive Returns): {self.df['target'].mean():.2%}")

    # ---- rolling centrality ------------------------------------------------
    def _rolling_centrality(self, returns_long):
        """
        Builds a correlation network using only the past 60 days of data and calculates
        each sector's PageRank. By sliding this window day by day, we ensure the model
        only knows what a real trader would know on that specific date.
        """
        cfg = self.cfg
        dcol = cfg["date_col"]
        wide = returns_long.pivot(index=dcol, columns="Sector", values="Return").sort_index()
        sectors = [s for s in cfg["sectors"] if s in wide.columns]
        w = cfg["centrality_window"]
        dates = wide.index

        records = []
        for i in range(w, len(dates)):
            # Grab the trailing window up to the current day
            window = wide.iloc[i - w:i + 1][sectors]   # trailing window incl. current day
            corr = window.corr()

            G = nx.Graph()
            G.add_nodes_from(sectors)

            # Connect sectors if they are highly correlated
            for a, b in itertools.combinations(sectors, 2):
                c = corr.loc[a, b]
                if pd.notna(c) and abs(c) > cfg["corr_threshold"]:
                    G.add_edge(a, b, weight=abs(c))

            if G.number_of_edges() > 0:
                pr = nx.pagerank(G, weight="weight")
            else:
                pr = {s: 1.0 / len(sectors) for s in sectors}

            for s in sectors:
                records.append({dcol: dates[i], "Sector": s, "pagerank": pr.get(s, 0.0)})

        return pd.DataFrame(records)

    # ---- purged + embargoed CV --------------------------------------------
    def _purged_splits(self):
        """
        Custom TimeSeriesSplit to prevent data leakage.
        We group by dates (not rows) so sectors from the same day stay together.
        We also 'embargo' (drop) the final few days of the training set to prevent
        the 5-day future target from bleeding into the test boundary.
        """
        cfg = self.cfg
        dates = np.sort(self.df[cfg["date_col"]].unique())
        date_series = self.df[cfg["date_col"]].values
        tscv = TimeSeriesSplit(n_splits=cfg["n_splits"])

        for tr_idx, te_idx in tscv.split(dates):
            train_dates = dates[tr_idx]
            test_dates = dates[te_idx]

            # Apply the embargo
            if cfg["embargo"] > 0 and len(train_dates) > cfg["embargo"]:
                train_dates = train_dates[:-cfg["embargo"]]

            tr_mask = np.isin(date_series, train_dates)
            te_mask = np.isin(date_series, test_dates)
            yield np.where(tr_mask)[0], np.where(te_mask)[0]

    # ---- generic sklearn evaluation ---------------------------------------
    def _evaluate(self, name, model_factory):
        X = self.df[self.X_cols].values
        y = self.df["target"].values
        accs, aucs, f1s = [], [], []

        for tr, te in self._purged_splits():
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])

            model = model_factory()
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)

            try:
                prob = model.predict_proba(Xte)[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
            except (AttributeError, ValueError):
                pass

            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="weighted"))

        self._report(name, accs, f1s, aucs)

    # ---- neural network ------
    def _evaluate_nn(self):
        X = self.df[self.X_cols].values
        y = self.df["target"].values
        accs, aucs, f1s = [], [], []

        # We use early stopping to prevent the network from overfitting to market noise
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        for tr, te in self._purged_splits():
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])

            # A relatively compact architecture since financial data is highly noisy
            model = Sequential([
                Dense(32, activation="relu", input_dim=Xtr.shape[1]),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

            # Reserving the chronological last 20% of the train fold for validation
            model.fit(Xtr, y[tr], epochs=100, batch_size=64, verbose=0,
                      validation_split=0.2, callbacks=[es])

            prob = model.predict(Xte, verbose=0).flatten()
            pred = (prob > 0.5).astype(int)

            accs.append(accuracy_score(y[te], pred))
            aucs.append(roc_auc_score(y[te], prob))
            f1s.append(f1_score(y[te], pred, average="weighted"))

        self._report("Neural Network", accs, f1s, aucs)

    @staticmethod
    def _report(name, accs, f1s, aucs):
        def ms(v):
            return f"{np.mean(v):.4f} (+/- {np.std(v):.4f})" if v else "n/a"
        print(f"\n  {name}")
        print(f"    Accuracy: {ms(accs)}")
        print(f"    F1 (wtd): {ms(f1s)}")
        print(f"    ROC-AUC:  {ms(aucs)}")

    # ---- orchestrator ------------------------------------------------------
    def run_models(self):
        self.load_and_prepare_data()
        print("\n--- 3. MODEL EVALUATION (purged + embargoed expanding-window CV) ---")

        # We include a naive baseline. If our ML models can't beat this, they aren't adding real value.
        self._evaluate("Naive baseline (most frequent)",
                       lambda: DummyClassifier(strategy="most_frequent"))
        self._evaluate("Logistic Regression (L2 / Ridge)",
                       lambda: LogisticRegression(penalty="l2", max_iter=1000, random_state=SEED))
        self._evaluate("Logistic Regression (L1 / Lasso)",
                       lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                                  max_iter=1000, random_state=SEED))
        self._evaluate_nn()
        print("\nDone. Compare every model against the naive baseline -- a model only "
              "adds value if it clears the base rate / 0.5 AUC by a robust margin.")


if __name__ == "__main__":
    # If running directly from the src/ folder, ensure data_dir points back to the root    pa = PredictiveAnalysis(data_dir="../data")
    pa = PredictiveAnalysis(data_dir="../data")
    pa.run_models()
