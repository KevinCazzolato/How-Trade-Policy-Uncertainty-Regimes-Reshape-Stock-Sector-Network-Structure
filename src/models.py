import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")


class HMMRegimeAnalysis:

    def __init__(self, smooth_window=10):
        try:
            self.base_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.smooth_window = smooth_window

    def load_and_prepare_data(self):
        master_path = self.data_dir / "master_dataset.csv"
        if not master_path.exists():
            raise FileNotFoundError(f"❌ Master dataset not found in {master_path}.")

        master = pd.read_csv(master_path, index_col=0, parse_dates=True)

        # Feature engineering
        master["log_tpu"] = np.log(master["tpu"])
        master["delta_log_tpu"] = master["log_tpu"].diff()

        features = ["log_tpu", "delta_log_tpu", "vix", "tpu"]
        self.master_hmm = master[features].dropna().copy()
        return self.master_hmm

    def check_stationarity(self):
        print("═" * 65 + "\nADF STATIONARITY TESTS\n" + "═" * 65)

        for name, series in [
            ("log(TPU) levels", self.master_hmm["log_tpu"]),
            ("Δlog(TPU) raw", self.master_hmm["delta_log_tpu"]),
            ("VIX levels", self.master_hmm["vix"]),
        ]:
            stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
            status = "I(0) ✓" if pval < 0.05 else "I(1) ✗  (near-unit-root)"
            print(f"  {name:25} | p={pval:.4f} | ADF={stat:>7.3f} | {status}")
        print("═" * 65 + "\n")

    def fit_hmm_robust(self, X_raw, k, name, n_restarts=30):
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        n, d = X.shape

        best_model, best_ll_mean = None, -np.inf
        for seed in range(n_restarts):
            m = GaussianHMM(
                n_components=k,
                covariance_type="full",
                n_iter=500,
                random_state=seed,
                tol=1e-6,
            )
            try:
                m.fit(X)
                if m.monitor_.converged and (ll_mean := m.score(X)) > best_ll_mean:
                    best_ll_mean, best_model = ll_mean, m
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError(f"no converged runs for [{name} k={k}]")

        ll_total = best_ll_mean * n
        p_count = k * (k - 1) + k * d + k * d * (d + 1) // 2
        aic = -2 * ll_total + 2 * p_count
        bic = -2 * ll_total + p_count * np.log(n)

        print(
            f"  [{name} | k={k}] LL/obs={best_ll_mean:.3f}  AIC={aic:,.1f}  BIC={bic:,.1f}"
        )
        return {
            "model": best_model,
            "scaler": scaler,
            "labels": best_model.predict(X),
            "probs": best_model.predict_proba(X),
            "bic": bic,
            "name": name,
            "k": k,
        }

    def run_analysis(self):
        self.load_and_prepare_data()
        self.check_stationarity()

        # Stima candidati
        candidates = {}
        print("Model estimation...")
        for k in [2, 3]:
            candidates[f"uni_k{k}"] = self.fit_hmm_robust(
                self.master_hmm[["log_tpu"]].values, k, "log(TPU) univariate"
            )
            candidates[f"bi_k{k}"] = self.fit_hmm_robust(
                self.master_hmm[["log_tpu", "vix"]].values, k, "log(TPU)+VIX bivariate"
            )

        res = candidates[min(candidates, key=lambda c: candidates[c]["bic"])]
        print(f"\n→ BIC-selected model: {res['name']} (k={res['k']})\n")

        # Ordinamento stati
        means_orig = (
            res["model"].means_[:, 0] * res["scaler"].scale_[0] + res["scaler"].mean_[0]
        )
        state_order = np.argsort(means_orig)
        low, high = state_order[0], state_order[-1]
        mid = state_order[1] if res["k"] == 3 else None

        def get_label(s):
            return "High-TPU" if s == high else ("Low-TPU" if s == low else "Mid-TPU")

        labels_raw = np.array([get_label(s) for s in res["labels"]])

        probs_df = pd.DataFrame(res["probs"], index=self.master_hmm.index)
        probs_smooth = (
            probs_df.rolling(window=self.smooth_window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        order_idx = [low, mid, high] if mid is not None else [low, high]
        labels_smooth = np.array(
            [
                (
                    "Low-TPU"
                    if s == 0
                    else ("Mid-TPU" if s == 1 and mid is not None else "High-TPU")
                )
                for s in np.argmax(probs_smooth[:, order_idx], axis=1)
            ]
        )

        regime_df = pd.DataFrame(
            {
                "regime_label": labels_smooth,
                "regime_label_raw": labels_raw,
                "prob_high_tpu": res["probs"][:, high],
                "prob_high_tpu_smooth": probs_smooth[:, high],
                "prob_mid_tpu_smooth": (
                    probs_smooth[:, mid]
                    if mid is not None
                    else np.zeros(len(self.master_hmm))
                ),
                "prob_low_tpu_smooth": probs_smooth[:, low],
                **{col: self.master_hmm[col].values for col in self.master_hmm.columns},
            },
            index=self.master_hmm.index,
        )

        # Salvataggio relativo in data/
        regime_df.to_csv(self.data_dir / "regime_labels.csv")

        # Stampa Diagnostiche Econometriche
        self._print_diagnostics(labels_smooth, res, get_label, means_orig, regime_df)

        # Generazione Plot Grafico
        self._plot_results(res, labels_smooth, probs_smooth, high, mid, low)

        return regime_df

    def _print_diagnostics(self, labels_smooth, res, get_label, means_orig, regime_df):
        print("═" * 60 + "\nREGIME DIAGNOSTICS (smoothed labels)\n" + "═" * 60)
        for reg, cnt in pd.Series(labels_smooth).value_counts().items():
            print(f"  {reg}: {cnt} days ({cnt/len(regime_df):.1%})")

        blocks = pd.Series(labels_smooth).ne(pd.Series(labels_smooth).shift()).cumsum()
        spell_stat = (
            pd.DataFrame({"label": labels_smooth, "block": blocks})
            .groupby("block")["label"]
            .agg(["count", "first"])
        )
        print("\nSpell durations:")
        for reg in spell_stat["first"].unique():
            subset = spell_stat[spell_stat["first"] == reg]["count"]
            print(f"  {reg:8}: avg {subset.mean():.1f} days  (max {subset.max()} days)")

        transmat = res["model"].transmat_
        exp_dur = 1.0 / (1.0 - np.diag(transmat))
        vals, vecs = np.linalg.eig(transmat.T)
        stat_dist = np.real(vecs[:, np.isclose(vals, 1)].ravel())
        stat_dist /= stat_dist.sum()

        print("── EMISSION MEANS (log(TPU) original scale) ──")
        for s in range(res["k"]):
            print(
                f"  State {s} [{get_label(s):10}]: log(TPU)={means_orig[s]:.4f} → TPU≈{np.exp(means_orig[s]):.1f}"
            )

        print("\n── TRANSITION MATRIX ──")
        for i in range(res["k"]):
            row = "  ".join(
                [
                    f"P({get_label(i)}→{get_label(j)})={transmat[i,j]:.4f}"
                    for j in range(res["k"])
                ]
            )
            print(f"  {row}")

        print("\n── LONG-RUN STATISTICS ──")
        for s in range(res["k"]):
            print(
                f"  {get_label(s):10}: E[dur]={exp_dur[s]:.1f}d  π={stat_dist[s]:.1%}"
            )

        # ── VIX STATS BY SMOOTHED REGIME ──
        print("\n" + "═" * 55 + "\nVIX STATS BY SMOOTHED REGIME\n" + "═" * 55)
        vix_stats = regime_df.groupby("regime_label")["vix"].agg(
            ["mean", "std", "count"]
        )
        print(
            vix_stats.reindex(
                [r for r in ["Low-TPU", "Mid-TPU", "High-TPU"] if r in vix_stats.index]
            ).to_string(
                formatters={
                    "mean": "{:,.2f}".format,
                    "std": "{:,.2f}".format,
                    "count": "{:,.0f}".format,
                }
            )
        )

        print(
            "\n" + "═" * 60 + "\n ARE REGIMES STRUCTURAL OR JUST A TREND?\n" + "═" * 60
        )
        spell_seq = regime_df.groupby(
            regime_df["regime_label"].ne(regime_df["regime_label"].shift()).cumsum()
        )["regime_label"].first()

        print("Number of distinct episodes per regime:")
        for reg in ["Low-TPU", "Mid-TPU", "High-TPU"]:
            print(f"  {reg}: {spell_seq.value_counts().get(reg, 0)} distinct episodes")

        r_seq = spell_seq.values
        rank = {"Low-TPU": 0, "Mid-TPU": 1, "High-TPU": 2}
        diffs = [rank[r_seq[i + 1]] - rank[r_seq[i]] for i in range(len(r_seq) - 1)]
        up = sum(1 for d in diffs if d > 0)
        down = sum(1 for d in diffs if d < 0)

        print(f"\nTransitions to higher regimes (↑): {up}")
        print(f"Transitions to lower regimes (↓): {down}")
        print(
            f"\n→ {'Structural regimes ✓ (bidirectional movement)' if down > 0 else 'WARNING: unidirectional trend only ✗'}"
        )
        print("═" * 60 + "\n")

    def _plot_results(self, res, labels_smooth, probs_smooth, high, mid, low):
        color_map = {"High-TPU": "red", "Mid-TPU": "orange", "Low-TPU": "steelblue"}
        high_mask = labels_smooth == "High-TPU"
        mid_mask = labels_smooth == "Mid-TPU"

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.suptitle(
            f"HMM Regime Detection — {res['name']} [k={res['k']}, BIC-selected]\n"
            f"Labels: argmax of {self.smooth_window}-day smoothed state probabilities",
            fontsize=13,
        )

        axes[0].plot(
            self.master_hmm.index, self.master_hmm["tpu"], lw=0.8, color="black"
        )
        axes[0].fill_between(
            self.master_hmm.index,
            0,
            self.master_hmm["tpu"].max(),
            where=high_mask,
            alpha=0.25,
            color="red",
            label="High-TPU",
        )
        axes[0].fill_between(
            self.master_hmm.index,
            0,
            self.master_hmm["tpu"].max(),
            where=mid_mask,
            alpha=0.15,
            color="orange",
            label="Mid-TPU",
        )
        axes[0].set(title="TPU Level + Regime Shading", yscale="log")
        axes[0].legend(fontsize=8)

        axes[1].plot(
            self.master_hmm.index,
            probs_smooth[:, high],
            color="red",
            lw=1.2,
            label="P(High-TPU)",
        )
        if mid is not None:
            axes[1].plot(
                self.master_hmm.index,
                probs_smooth[:, mid],
                color="orange",
                lw=1.2,
                label="P(Mid-TPU)",
            )
        axes[1].plot(
            self.master_hmm.index,
            probs_smooth[:, low],
            color="steelblue",
            lw=1.2,
            label="P(Low-TPU)",
        )
        axes[1].axhline(0.5, ls="--", lw=0.8, color="gray")
        axes[1].set(
            ylim=(0, 1), title=f"Smoothed State Probabilities (MA{self.smooth_window})"
        )
        axes[1].legend(fontsize=8)

        axes[2].bar(
            self.master_hmm.index,
            self.master_hmm["delta_log_tpu"],
            color=[color_map.get(l, "gray") for l in labels_smooth],
            width=1,
            alpha=0.7,
        )
        axes[2].axhline(0, lw=0.5, color="black")
        axes[2].set_title("Δlog(TPU) colored by smoothed regime")

        axes[3].plot(
            self.master_hmm.index, self.master_hmm["vix"], lw=0.8, color="green"
        )
        axes[3].fill_between(
            self.master_hmm.index,
            0,
            self.master_hmm["vix"].max(),
            where=high_mask,
            alpha=0.2,
            color="red",
        )
        axes[3].set_title("VIX + High-TPU overlay")

        plt.tight_layout()
        plt.savefig(self.data_dir / "hmm_regime_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to: {self.data_dir / 'hmm_regime_plot.png'}")


class HMM_TVTP:

    TARIFF_EVENTS = [
        ("2025-01-20", "Trump inauguration"),
        ("2025-02-01", "25% tariffs CAN/MEX"),
        ("2025-04-02", "Liberation Day"),
        ("2025-04-09", "90-day pause"),
        ("2025-05-12", "US-China truce"),
    ]

    def __init__(self):
        try:
            self.base_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.out_dir = self.data_dir / "phase_2"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        master = pd.read_csv(
            self.data_dir / "master_dataset.csv",
            index_col=0,
            parse_dates=True,
        )

        baseline = pd.read_csv(
            self.data_dir / "regime_labels.csv",
            index_col=0,
            parse_dates=True,
        )

        df = master.assign(
            log_tpu=np.log(master["tpu"]),
            log_vix=np.log(master["vix"]),
        ).dropna(subset=["log_tpu", "log_vix"])

        df = df.loc[df.index.intersection(baseline.index)]

        df["log_tpu_z"] = (df["log_tpu"] - df["log_tpu"].mean()) / df["log_tpu"].std()
        df["log_vix_z"] = (df["log_vix"] - df["log_vix"].mean()) / df["log_vix"].std()

        self.df = df
        self.baseline = baseline

        self.OBS = df["log_tpu_z"].values
        self.COND = df["log_vix_z"].values
        self.T = len(df)

        return df

    @staticmethod
    def transmat_t(a01, b01, a10, b10, z):
        p01 = expit(a01 + b01 * z)
        p10 = expit(a10 + b10 * z)
        return np.array([[1 - p01, p01], [p10, 1 - p10]])

    def forward_backward(self, obs, cond, mu, sigma, a01, b01, a10, b10):
        emit = np.column_stack([norm.pdf(obs, mu[k], sigma[k]) for k in range(2)])
        emit = np.clip(emit, 1e-300, None)

        fwd = np.zeros((self.T, 2))
        scales = np.zeros(self.T)

        fwd[0] = 0.5 * emit[0]
        scales[0] = fwd[0].sum() or 1e-300
        fwd[0] /= scales[0]

        for t in range(1, self.T):
            A = self.transmat_t(a01, b01, a10, b10, cond[t - 1])
            fwd[t] = (fwd[t - 1] @ A) * emit[t]
            scales[t] = fwd[t].sum() or 1e-300
            fwd[t] /= scales[t]

        bwd = np.ones((self.T, 2))
        for t in range(self.T - 2, -1, -1):
            A = self.transmat_t(a01, b01, a10, b10, cond[t])
            bwd[t] = (A * emit[t + 1] * bwd[t + 1]).sum(axis=1)
            bwd[t] /= bwd[t].sum() or 1e-300

        gamma = fwd * bwd
        gamma /= gamma.sum(axis=1, keepdims=True)

        return np.sum(np.log(scales)), gamma

    def make_nll(self, fix_betas=False):
        def nll(p, obs, cond):
            mu0, ls0, mu1, ls1 = p[:4]

            if fix_betas:
                a01, b01, a10, b10 = p[4], 0.0, p[5], 0.0
            else:
                a01, b01, a10, b10 = p[4:]

            s0 = np.exp(ls0)
            s1 = np.exp(ls1)

            if s0 < 1e-6 or s1 < 1e-6:
                return 1e10

            ll, _ = self.forward_backward(
                obs, cond, [mu0, mu1], [s0, s1], a01, b01, a10, b10
            )
            return -ll

        return nll

    def fit(self):
        bl = self.baseline.reindex(self.df.index)
        high = bl["regime_label"] == "High-TPU"

        mu0 = self.OBS[~high].mean()
        mu1 = self.OBS[high].mean()
        s0 = max(self.OBS[~high].std(), 0.05)
        s1 = max(self.OBS[high].std(), 0.05)

        logit = lambda p: np.log(p / (1 - p))

        opt_kw = dict(
            method="L-BFGS-B",
            options=dict(maxiter=3000, ftol=1e-12, gtol=1e-8),
        )

        self.base = minimize(
            self.make_nll(True),
            [mu0, np.log(s0), mu1, np.log(s1), logit(0.05), logit(0.05)],
            args=(self.OBS, self.COND),
            **opt_kw,
        )

        self.model = minimize(
            self.make_nll(False),
            [mu0, np.log(s0), mu1, np.log(s1), logit(0.05), 0, logit(0.05), 0],
            args=(self.OBS, self.COND),
            **opt_kw,
        )

        # Estrazione parametri stimati come attributi d'istanza
        self.ll_base, self.ll_tvtp = -self.base.fun, -self.model.fun
        (
            self.mu0,
            self.ls0,
            self.mu1,
            self.ls1,
            self.a01,
            self.b01,
            self.a10,
            self.b10,
        ) = self.model.x
        self.sigma0, self.sigma1 = np.exp(self.ls0), np.exp(self.ls1)

        # Probabilità filtrate/smussate sul modello pieno (TVTP)
        _, self.gamma = self.forward_backward(
            self.OBS,
            self.COND,
            [self.mu0, self.mu1],
            [self.sigma0, self.sigma1],
            self.a01,
            self.b01,
            self.a10,
            self.b10,
        )
        self.prob_high = self.gamma[:, 1]
        self.prob_low = self.gamma[:, 0]
        self.labels = np.where(self.prob_high > 0.5, "High-TPU", "Non-High-TPU")

        self.p01 = expit(self.a01 + self.b01 * self.COND)
        self.p10 = expit(self.a10 + self.b10 * self.COND)

    def _print_diagnostics(self):
        T = self.T
        aic = lambda ll, k: -2 * ll + 2 * k
        bic = lambda ll, k: -2 * ll + k * np.log(T)

        lr_stat = 2 * (self.ll_tvtp - self.ll_base)
        lr_pval = 1 - chi2.cdf(lr_stat, df=2)
        conclusion = (
            "REJECT H0 → VIX drives transitions"
            if lr_pval < 0.05
            else "FAIL TO REJECT H0"
        )

        print(
            "═" * 60
            + f"\n{'Model':<22} {'LL':>10} {'k':>4} {'AIC':>10} {'BIC':>10}\n"
            + "-" * 60
        )
        print(
            f"{'Restricted (β=0)':<22} {self.ll_base:>10,.1f} {6:>4} "
            f"{aic(self.ll_base, 6):>10,.1f} {bic(self.ll_base, 6):>10,.1f}"
        )
        print(
            f"{'TVTP-HMM':<22} {self.ll_tvtp:>10,.1f} {8:>4} "
            f"{aic(self.ll_tvtp, 8):>10,.1f} {bic(self.ll_tvtp, 8):>10,.1f}"
        )
        print(
            f"\nLR test: LR={lr_stat:.3f} p={lr_pval:.4e} → {conclusion}\n" + "═" * 60
        )

        log_tpu_mean, log_tpu_std = self.df["log_tpu"].mean(), self.df["log_tpu"].std()

        tvtp_df = pd.DataFrame(
            {
                "regime_label": self.labels,
                "prob_high_tpu": self.prob_high,
                "prob_nonhigh": self.prob_low,
                "p01_entry": self.p01,
                "p10_exit": self.p10,
                "tpu_level": self.df["tpu"].values,
                "log_tpu_z": self.OBS,
                "log_vix_z": self.COND,
                "vix": self.df["vix"].values,
            },
            index=self.df.index,
        )
        tvtp_df.to_csv(self.out_dir / "tvtp_regime_labels.csv")
        self.tvtp_df = tvtp_df

        with open(self.out_dir / "tvtp_diagnostics.txt", "w") as f:
            f.write(
                f"═══ TVTP-HMM DIAGNOSTICS ═══\nObservations: {T}\n"
                f"Non-High: μ={self.mu0*log_tpu_std+log_tpu_mean:.4f} σ={self.sigma0*log_tpu_std:.4f}\n"
                f"High-TPU: μ={self.mu1*log_tpu_std+log_tpu_mean:.4f} σ={self.sigma1*log_tpu_std:.4f}\n"
                f"α01={self.a01:.4f} β01={self.b01:.4f} | α10={self.a10:.4f} β10={self.b10:.4f}\n"
                f"LR Stat={lr_stat:.3f} p={lr_pval:.4e} ({conclusion})"
            )

    def _plot_results(self):
        df = self.df
        bl = self.baseline.reindex(df.index)
        high_mask_tvtp = self.labels == "High-TPU"
        RED, BLUE, GREEN = "#E24B4A", "#378ADD", "#1D9E75"

        fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
        fig.suptitle(
            "TVTP-HMM — Time-Varying Transition Probabilities (Filardo 1994)",
            fontsize=12,
        )

        for ax, col, title, scale in zip(
            [axes[0], axes[4]],
            ["tpu", "vix"],
            ["A — TPU Index", "E — VIX"],
            ["log", "linear"],
        ):
            ax.plot(df.index, df[col], color="#444" if col == "tpu" else BLUE, lw=0.7)
            ax.fill_between(
                df.index, 0, df[col].max(), where=high_mask_tvtp, alpha=0.2, color=RED
            )
            ax.set(yscale=scale, title=title)

        axes[1].plot(
            df.index,
            bl["prob_high_tpu_smooth"].reindex(df.index),
            color=BLUE,
            lw=0.8,
            label="Baseline k=3",
        )
        axes[1].fill_between(
            df.index, 0, self.prob_high, color=RED, alpha=0.45, label="TVTP k=2"
        )
        axes[1].axhline(0.5, color="#888", lw=0.8, ls="--")
        axes[1].set(ylim=(0, 1), title="B — Smoothed P(High-TPU): TVTP vs Baseline")
        axes[1].legend(fontsize=8)

        for date_str, label in self.TARIFF_EVENTS:
            d = pd.Timestamp(date_str)
            if df.index[0] <= d <= df.index[-1]:
                for ax in axes[:2]:
                    ax.axvline(d, color=BLUE, lw=0.9, ls=":", alpha=0.7)
                axes[1].text(
                    d,
                    0.52,
                    label,
                    fontsize=5.5,
                    color=BLUE,
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
                )

        for ax, series, color, title in zip(
            [axes[2], axes[3]],
            [self.p01, self.p10],
            [RED, GREEN],
            [
                r"C — Time-Varying Entry Probability $p_{01}(t)$",
                r"D — Time-Varying Exit Probability $p_{10}(t)$",
            ],
        ):
            ax.plot(df.index, series, color=color, lw=0.9)
            ax.set(ylim=(0, min(series.max() * 1.1, 1)), title=title)

        axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[4].xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()
        plt.savefig(self.out_dir / "tvtp_plots.png", dpi=150)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        vix_grid_z = np.linspace(self.COND.min(), self.COND.max(), 300)
        vix_grid = np.exp(vix_grid_z * df["log_vix"].std() + df["log_vix"].mean())

        for ax, prob_grid, color, title in zip(
            axes2,
            [
                expit(self.a01 + self.b01 * vix_grid_z),
                expit(self.a10 + self.b10 * vix_grid_z),
            ],
            [RED, GREEN],
            ["VIX → Entry into High-TPU", "VIX → Exit from High-TPU"],
        ):
            ax.plot(vix_grid, prob_grid, color=color, lw=2)
            ax.axhline(0.5, color="#888", lw=0.8, ls="--")
            ax.set(xlabel="VIX", title=title)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.out_dir / "tvtp_vix_effect.png", dpi=150)
        plt.close(fig2)

    def run_analysis(self):
        self.load_and_prepare_data()
        self.fit()
        self._print_diagnostics()
        self._plot_results()


class ClusteringBenchmark:
    """
    Phase 2 — non-temporal clustering benchmarks (K-Means, Hierarchical) against
    the HMM regime structure, plus a chi-square event-window validation against
    the 2025 tariff timeline.
    """

    def __init__(self):
        try:
            self.base_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.out_dir = self.data_dir / "phase_2"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        master = pd.read_csv(
            self.data_dir / "master_dataset.csv", index_col=0, parse_dates=True
        )
        regime_df = pd.read_csv(
            self.data_dir / "regime_labels.csv", index_col=0, parse_dates=True
        )

        master["log_tpu"] = np.log(master["tpu"])
        master_hmm = master[["log_tpu", "vix"]].dropna()

        common_idx = master_hmm.index.intersection(regime_df.index)
        master_hmm, regime_df = (
            master_hmm.loc[common_idx],
            regime_df.loc[common_idx],
        )

        self.master = master
        self.master_hmm = master_hmm
        self.regime_df = regime_df
        return master_hmm, regime_df

    def fit_clustering_benchmarks(self):
        print(f"\n{'═'*60}\nNON-TEMPORAL CLUSTERING BENCHMARKS\n{'═'*60}")

        X_scaled = StandardScaler().fit_transform(
            self.master_hmm[["log_tpu", "vix"]].values
        )
        best_k = len(self.regime_df["regime_label"].unique())
        hmm_labels = self.regime_df["regime_label"].values

        kmeans_labels = KMeans(
            n_clusters=best_k, random_state=42, n_init=10
        ).fit_predict(X_scaled)
        hc_labels = AgglomerativeClustering(
            n_clusters=best_k, linkage="ward"
        ).fit_predict(X_scaled)

        print(
            f"Silhouette Scores:\n"
            f"  K-Means (k={best_k}): {silhouette_score(X_scaled, kmeans_labels):.3f}"
        )
        print(
            f"  Hierarchical (k={best_k}): {silhouette_score(X_scaled, hc_labels):.3f}"
        )

        print(f"\nAgreement with HMM Regimes (ARI | NMI):")
        ari_km = adjusted_rand_score(hmm_labels, kmeans_labels)
        nmi_km = normalized_mutual_info_score(hmm_labels, kmeans_labels)
        ari_hc = adjusted_rand_score(hmm_labels, hc_labels)
        nmi_hc = normalized_mutual_info_score(hmm_labels, hc_labels)
        print(f"  K-Means:      {ari_km:.3f} | {nmi_km:.3f}")
        print(f"  Hierarchical: {ari_hc:.3f} | {nmi_hc:.3f}")

        if ari_km < 0.6:
            print(
                "\n→ ECONOMETRIC INSIGHT: The low ARI shows that the Markov dependence of the HMM matters."
            )

        self.best_k = best_k
        self.kmeans_labels = kmeans_labels
        self.hc_labels = hc_labels
        self.cluster_scores = {
            "silhouette_kmeans": silhouette_score(X_scaled, kmeans_labels),
            "silhouette_hc": silhouette_score(X_scaled, hc_labels),
            "ari_kmeans": ari_km,
            "nmi_kmeans": nmi_km,
            "ari_hc": ari_hc,
            "nmi_hc": nmi_hc,
        }
        return kmeans_labels, hc_labels

    def validate_tariff_events(self):
        print(f"\n{'═'*60}\nEMPIRICAL EVENT VALIDATION (2025 TARIFFS)\n{'═'*60}")

        regime_df = self.regime_df
        masks = [
            (regime_df.index >= pd.Timestamp(d) - pd.Timedelta(days=2))
            & (regime_df.index <= pd.Timestamp(d) + pd.Timedelta(days=10))
            for d, _ in HMM_TVTP.TARIFF_EVENTS
        ]
        regime_df["is_event_window"] = np.logical_or.reduce(masks)
        regime_df["is_high_tpu"] = regime_df["regime_label"] == "High-TPU"

        contingency_table = pd.crosstab(
            regime_df["is_event_window"], regime_df["is_high_tpu"]
        )
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)

        print("Contingency Table:\n", contingency_table)
        print(f"\nPearson Chi-Square: Chi2 = {chi2_stat:.2f} | P-value = {p_val:.4e}")
        conclusion = (
            "REJECT H0 at level 5%" if p_val < 0.05 else "FAIL TO REJECT H0 at level 5%"
        )
        print(f"\n→ {conclusion}")

        self.chi2_stat, self.chi2_pval = chi2_stat, p_val
        return chi2_stat, p_val

    def _plot_results(self):

        master_hmm = self.master_hmm.copy()
        master_hmm["kmeans_cluster"] = self.kmeans_labels
        master_hmm["hc_cluster"] = self.hc_labels

        plot_df = self.master.loc[master_hmm.index].copy()
        plot_df["high_hmm"] = self.regime_df["regime_label"] == "High-TPU"
        plot_df["high_kmeans"] = (
            master_hmm["kmeans_cluster"]
            == master_hmm.groupby("kmeans_cluster")["log_tpu"].mean().idxmax()
        )
        plot_df["high_hc"] = (
            master_hmm["hc_cluster"]
            == master_hmm.groupby("hc_cluster")["log_tpu"].mean().idxmax()
        )

        panels = [
            {
                "col": "high_hmm",
                "color": "#E24B4A",
                "title": "A — Markov Switching HMM (Includes Time-Dependency)",
                "lbl": "HMM High-TPU Regime",
            },
            {
                "col": "high_kmeans",
                "color": "#8E44AD",
                "title": f"B — K-Means Static Clustering Benchmark (k={self.best_k})",
                "lbl": "K-Means High Cluster",
            },
            {
                "col": "high_hc",
                "color": "#E67E22",
                "title": f"C — Agglomerative Hierarchical Benchmark (Ward, k={self.best_k})",
                "lbl": "Hierarchical High Cluster",
            },
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
        fig.suptitle(
            "Regime Identification: Markovian HMM vs. Non-Temporal Clustering Benchmarks",
            fontsize=14,
            fontweight="bold",
            y=0.96,
        )

        for ax, p in zip(axes, panels):
            ax.plot(
                plot_df.index, plot_df["tpu"], color="#444", lw=0.8, label="TPU Index"
            )
            ax.fill_between(
                plot_df.index,
                0,
                plot_df["tpu"].max(),
                where=plot_df[p["col"]],
                alpha=0.22,
                color=p["color"],
                label=p["lbl"],
            )
            ax.set(yscale="log", title=p["title"])
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.2, ls="--")

            for date_str, label in HMM_TVTP.TARIFF_EVENTS:
                ev_date = pd.Timestamp(date_str)
                if plot_df.index[0] <= ev_date <= plot_df.index[-1]:
                    ax.axvline(ev_date, color="#378ADD", lw=1.0, ls=":", alpha=0.8)

        for date_str, label in HMM_TVTP.TARIFF_EVENTS:
            ev_date = pd.Timestamp(date_str)
            if plot_df.index[0] <= ev_date <= plot_df.index[-1]:
                axes[0].text(
                    ev_date,
                    plot_df["tpu"].max() * 0.4,
                    label,
                    fontsize=7,
                    color="#378ADD",
                    rotation=90,
                    va="top",
                    ha="right",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="#378ADD",
                        alpha=0.8,
                        lw=0.5,
                    ),
                )

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        plt.tight_layout()

        output_path = self.out_dir / "hmm_vs_all_benchmarks.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Grafico salvato in: {output_path}")

    def run_analysis(self):
        self.load_and_prepare_data()
        self.fit_clustering_benchmarks()
        self.validate_tariff_events()
        self._plot_results()


if __name__ == "__main__":
    try:
        hmm = HMMRegimeAnalysis()
        hmm.run_analysis()
    except Exception as e:
        print(f"error HMM: {e}")

    try:
        tvtp = HMM_TVTP()
        tvtp.run_analysis()
    except Exception as e:
        print(f"error TVTP: {e}")

    try:
        bench = ClusteringBenchmark()
        bench.run_analysis()
    except Exception as e:
        print(f"error ClusteringBenchmark: {e}")
