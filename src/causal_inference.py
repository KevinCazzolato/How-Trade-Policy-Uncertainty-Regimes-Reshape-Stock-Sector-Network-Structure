# causal Inference Part   

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from dowhy import CausalModel
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings("ignore")

class CausalInferenceAnalysis:
    """
    Phase 3 causal inference pipeline.

    Inputs (read from data/):
        - master_dataset.csv         sector returns + macro controls
        - phase_2/tvtp_regime_labels.csv   TVTP regime labels

    Outputs (saved to data/phase_3/):
        - dag.png              causal DAG visualization
        - purged_returns.csv   sector returns after backdoor adjustment
    """

    # 11 US Select Sector SPDR ETFs (defined in data_acquisition.py)
    SECTORS = ["XLK", "XLI", "XLY", "XLF", "XLV", "XLE",
               "XLB", "XLRE", "XLU", "XLP", "XLC"]

    # Confounders for backdoor adjustment (from proposal Section 4a)
    CONFOUNDERS = ["vix", "spread_10y2y"]

    def __init__(self):
        # Locate project root
        try:
            self.base_dir = Path(__file__).resolve().parents[1]
        except NameError:
            self.base_dir = Path.cwd()

        self.data_dir = self.base_dir / "data"
        self.out_dir = self.data_dir / "phase_3"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    
    # Load and align data
    
    def load_data(self):
        """Read master + TVTP regime labels, inner-join on date."""
        print("Loading data...")

        master = pd.read_csv(
            self.data_dir / "master_dataset.csv",
            index_col=0, parse_dates=True,
        )
        
        # Undo the 1-day lag for causal modeling
        master["vix"] = master["vix"].shift(-1)
        master["spread_10y2y"] = master["spread_10y2y"].shift(-1)

        tvtp = pd.read_csv(
            self.data_dir / "phase_2" / "tvtp_regime_labels.csv",
            index_col=0, parse_dates=True,
        )

        # Keep only the columns needed
        master = master[self.SECTORS + self.CONFOUNDERS]
        regime = tvtp[["regime_label"]]

        # Inner join on date 
        df = master.join(regime, how="inner").dropna()

        # Binary version of the regime for the DoWhy
        df["tpu_regime"] = (df["regime_label"] == "High-TPU").astype(int)

        print(f"  Combined dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
        print(f"  Regime distribution:")
        for name, count in df["regime_label"].value_counts().items():
            print(f"    {name:15s}: {count:5d} days ({count/len(df):.1%})")

        self.df = df
        return df

    
    # DAG and verify backdoor adjustment
    
    def build_dag(self):
        """Construct the causal DAG and check the backdoor criterion."""
        print("\nSTEP 2: Build DAG and identifying causal effect...")

        # Use XLK as outcome for identification.
        # The same DAG structure applies to all 11 sectors symmetrically.
        model = CausalModel(
            data=self.df,
            treatment="tpu_regime",
            outcome="XLK",
            common_causes=self.CONFOUNDERS,
        )

        # Identify the causal effect (verifies backdoor criterion)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        print("\n  Identified estimand:")
        print("  ")
        print(estimand)

        # Save the DAG to a file
        self._plot_dag()

        self.causal_model = model
        return estimand

    def _plot_dag(self):
        """Draw the DAG: confounders -> {treatment, outcome}, treatment -> outcome."""
        G = nx.DiGraph()
        edges = [
            ("VIX", "TPU regime"),
            ("Rate spread (10y-2y)", "TPU regime"),
            ("VIX", "Sector return"),
            ("Rate spread (10y-2y)", "Sector return"),
            ("TPU regime", "Sector return"),
        ]
        G.add_edges_from(edges)

        # confounders to left, treatment to center and outcome to the right
        pos = {
            "VIX": (-1.5, 1),
            "Rate spread (10y-2y)": (-1.5, -1),
            "TPU regime": (0, 0),
            "Sector return": (1.5, 0),
        }

        node_colors = ["lightblue", "lightblue", "orange", "lightgreen"]

        fig, ax = plt.subplots(figsize=(10, 5))
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=4500,
            font_size=10, font_weight="bold",
            arrows=True, arrowsize=25,
            edge_color="gray", width=1.5,
            ax=ax,
        )
        ax.set_title(
            "Causal DAG \n"
            "VIX and rate spread block all backdoor paths from TPU to sector returns",
            fontsize=11,
        )
        ax.margins(0.15)
        plt.tight_layout()

        out_path = self.out_dir / "dag.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  DAG saved: {out_path}")

    
    # Backdoor adjustment by partialling-out
    
    def apply_backdoor_adjustment(self):
        """
        For each sector, regress its return on the confounders
        We keep the residuals as the VIX/rates-purged returns
        downstream network analysis uses these residuals
        """
        print("\nbackdoor adjustment using (partialling out VIX + spread)")

        # design matrix (with constant) once for all sectors
        X = sm.add_constant(self.df[self.CONFOUNDERS])

        purged_returns = pd.DataFrame(index=self.df.index)
        r2_scores = {}

        for sector in self.SECTORS:
            y = self.df[sector]
            ols = sm.OLS(y, X).fit()
            purged_returns[sector] = ols.resid
            r2_scores[sector] = ols.rsquared

        
        purged_returns["regime_label"] = self.df["regime_label"]

        # variance absorbed by VIX + spread per sector?
        print("\n  Confounder R2 per sector (higher  = more variance)")
        for sector, r2 in sorted(r2_scores.items(), key=lambda x: -x[1]):
            print(f"    {sector}: {r2:.3f}")

        out_path = self.out_dir / "purged_returns.csv"
        purged_returns.to_csv(out_path)
        print(f"\n  Purged returns saved: {out_path}")

        self.purged_returns = purged_returns
        return purged_returns


    # correlation networks per regime
    def compute_correlations(self):
        print("4. correlations per regime")

        high = self._regime_returns("High-TPU")
        low = self._regime_returns("Non-High-TPU")

        self.corr_high = high.corr()
        self.corr_low = low.corr()
        self.pcorr_high = self._partial_corr(high)
        self.pcorr_low = self._partial_corr(low)

        # save matrices
        self.corr_high.to_csv(self.out_dir / "corr_high.csv")
        self.corr_low.to_csv(self.out_dir / "corr_low.csv")
        self.pcorr_high.to_csv(self.out_dir / "pcorr_high.csv")
        self.pcorr_low.to_csv(self.out_dir / "pcorr_low.csv")

        # plots
        self._plot_corr(self.corr_low, self.corr_high, "Pearson", "corr.png")
        self._plot_corr(self.pcorr_low, self.pcorr_high, "Partial", "pcorr.png")

    def _regime_returns(self, regime):
        mask = self.purged_returns["regime_label"] == regime
        return self.purged_returns.loc[mask, self.SECTORS]

    def _partial_corr(self, df):
        # inverse of correlation matrix gives partial correlations
        corr = df.corr().values
        inv = np.linalg.inv(corr)
        d = np.sqrt(np.diag(inv))
        pcorr = -inv / np.outer(d, d)
        np.fill_diagonal(pcorr, 1.0)
        return pd.DataFrame(pcorr, index=df.columns, columns=df.columns)

    def _plot_corr(self, low, high, kind, filename):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(low, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title(f"{kind} corr — Non-High-TPU")
        sns.heatmap(high, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title(f"{kind} corr — High-TPU")
        plt.tight_layout()
        plt.savefig(self.out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # pairwise Granger causality
    def run_granger(self, lag=5):
        print(f"5. granger tests (lag={lag}, ~1 trading week)")

        high = self._regime_returns("High-TPU")
        low = self._regime_returns("Non-High-TPU")

        self.granger_high = self._granger_matrix(high, lag)
        self.granger_low = self._granger_matrix(low, lag)

        self.granger_high.to_csv(self.out_dir / "granger_pvals_high.csv")
        self.granger_low.to_csv(self.out_dir / "granger_pvals_low.csv")

    def _granger_matrix(self, df, lag):
        pvals = pd.DataFrame(np.nan, index=self.SECTORS, columns=self.SECTORS)
        fails = 0
        for i in self.SECTORS:
            for j in self.SECTORS:
                if i == j:
                    continue
                try:
                    data = df[[j, i]].dropna().values
                    result = grangercausalitytests(data, maxlag=lag, verbose=False)
                    pvals.loc[i, j] = result[lag][0]["ssr_ftest"][1]
                except Exception:
                    pvals.loc[i, j] = np.nan
                    fails += 1
        if fails > 0:
            print(f"   {fails} pairs failed")
        return pvals

    # BH-FDR correction
    def apply_fdr(self, q=0.05):
        print(f"6. BH-FDR (q={q})")

        self.edges_high = self._fdr_edges(self.granger_high, q)
        self.edges_low = self._fdr_edges(self.granger_low, q)

        self.edges_high.astype(int).to_csv(self.out_dir / "edges_high.csv")
        self.edges_low.astype(int).to_csv(self.out_dir / "edges_low.csv")

        # plot p-values before/after
        self._plot_granger()

    def _fdr_edges(self, pvals, q):
        # flatten the p-values matrix (taking out nan)
        mask = ~pvals.isna().values
        flat = pvals.values[mask]
        reject, _, _, _ = multipletests(flat, alpha=q, method="fdr_bh")

        sig = pd.DataFrame(False, index=pvals.index, columns=pvals.columns)
        sig.values[mask] = reject
        return sig

    def _plot_granger(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # raw p-values
        sns.heatmap(self.granger_low, cmap="rocket_r", vmin=0, vmax=0.1,
                    annot=True, fmt=".3f", ax=axes[0, 0],
                    annot_kws={"size": 7})
        axes[0, 0].set_title("p-values — Non-High-TPU")

        sns.heatmap(self.granger_high, cmap="rocket_r", vmin=0, vmax=0.1,
                    annot=True, fmt=".3f", ax=axes[0, 1],
                    annot_kws={"size": 7})
        axes[0, 1].set_title("p-values — High-TPU")

        # after FDR
        sns.heatmap(self.edges_low.astype(int), cmap="Greens",
                    vmin=0, vmax=1, cbar=False, ax=axes[1, 0])
        axes[1, 0].set_title("BH-FDR edges — Non-High-TPU")

        sns.heatmap(self.edges_high.astype(int), cmap="Greens",
                    vmin=0, vmax=1, cbar=False, ax=axes[1, 1])
        axes[1, 1].set_title("BH-FDR edges — High-TPU")

        plt.tight_layout()
        plt.savefig(self.out_dir / "granger.png", dpi=150, bbox_inches="tight")
        plt.close()

    # network metrics + plots
    def network_metrics(self):
        print("7. networks and centrality")

        G_high = self._build_graph(self.edges_high)
        G_low = self._build_graph(self.edges_low)

        self.metrics_high = self._metrics(G_high)
        self.metrics_low = self._metrics(G_low)

        self.metrics_high.to_csv(self.out_dir / "metrics_high.csv")
        self.metrics_low.to_csv(self.out_dir / "metrics_low.csv")

        # global summary per regime
        avg_c_low = nx.average_clustering(G_low.to_undirected())
        avg_c_high = nx.average_clustering(G_high.to_undirected())

        print(f"   Non-High-TPU: {G_low.number_of_edges()} edges, "
            f"density={nx.density(G_low):.3f}, avg_clustering={avg_c_low:.3f}")
        print(f"   High-TPU:     {G_high.number_of_edges()} edges, "
            f"density={nx.density(G_high):.3f}, avg_clustering={avg_c_high:.3f}")

        self._plot_networks(G_low, G_high)
        self.G_high = G_high
        self.G_low = G_low

    def _build_graph(self, edges_matrix):
        G = nx.DiGraph()
        G.add_nodes_from(self.SECTORS)
        for i in self.SECTORS:
            for j in self.SECTORS:
                if edges_matrix.loc[i, j]:
                    G.add_edge(i, j)
        return G

    def _metrics(self, G):
        m = pd.DataFrame(index=self.SECTORS)
        m["in_degree"] = pd.Series(dict(G.in_degree()))
        m["out_degree"] = pd.Series(dict(G.out_degree()))
        m["betweenness"] = pd.Series(nx.betweenness_centrality(G))
        m["pagerank"] = pd.Series(nx.pagerank(G, alpha=0.85))
        m["clustering"] = pd.Series(nx.clustering(G.to_undirected()))
        return m.round(3)

    def _plot_networks(self, G_low, G_high):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for ax, G, title in [(axes[0], G_low, "Non-High-TPU"),
                            (axes[1], G_high, "High-TPU")]:
            pos = nx.circular_layout(G)
            # node size grows with out-degree (hubness)
            sizes = [400 + 250 * G.out_degree(n) for n in G.nodes()]
            nx.draw(G, pos, ax=ax, with_labels=True,
                    node_color="lightblue", node_size=sizes,
                    font_size=10, font_weight="bold",
                    arrows=True, arrowsize=15,
                    edge_color="gray", width=0.8,
                    connectionstyle="arc3,rad=0.1")
            ax.set_title(f"granger network — {title}\n"
                        f"{G.number_of_edges()} edges, "
                        f"density={nx.density(G):.3f}")

        plt.tight_layout()
        plt.savefig(self.out_dir / "networks.png", dpi=150, bbox_inches="tight")
        plt.close()


    # Johansen cointegration + VECM per regime
    def run_johansen_vecm(self, lag_diff=1):
        print(f"8. johansen + VECM (lag diff={lag_diff})")

        # cointegration needs non-stationary series using log prices not returns
        log_prices = pd.read_csv(
            self.data_dir / "log_prices.csv",
            index_col=0, parse_dates=True,
        )

        # align with regime labels
        regime = self.purged_returns[["regime_label"]]
        prices = log_prices[self.SECTORS].join(regime, how="inner").dropna()

        # split by regime
        high = prices[prices["regime_label"] == "High-TPU"][self.SECTORS]
        low = prices[prices["regime_label"] == "Non-High-TPU"][self.SECTORS]

        self.coint_high, self.vecm_high = self._coint_and_vecm(high, "High-TPU", lag_diff)
        self.coint_low, self.vecm_low = self._coint_and_vecm(low, "Non-High-TPU", lag_diff)

        # save and plot
        self._save_coint_results()
        self._plot_adjustment_speeds()


    # lag sensitivity check for cointegrating rank from 1 to 3
    def lag_sensitivity_check(self, lags=(1, 2, 3)):
        print(f"   lag sensitivity check (lags={list(lags)})")

        log_prices = pd.read_csv(
            self.data_dir / "log_prices.csv",
            index_col=0, parse_dates=True,
        )
        regime = self.purged_returns[["regime_label"]]
        prices = log_prices[self.SECTORS].join(regime, how="inner").dropna()

        high = prices[prices["regime_label"] == "High-TPU"][self.SECTORS]
        low = prices[prices["regime_label"] == "Non-High-TPU"][self.SECTORS]

        results = []
        for lag in lags:
            joh_low = coint_johansen(low.values, det_order=0, k_ar_diff=lag)
            joh_high = coint_johansen(high.values, det_order=0, k_ar_diff=lag)
            rank_low = int((joh_low.lr1 > joh_low.cvt[:, 1]).sum())
            rank_high = int((joh_high.lr1 > joh_high.cvt[:, 1]).sum())
            results.append({"lag": lag, "rank_low": rank_low, "rank_high": rank_high})

        self.lag_sensitivity = pd.DataFrame(results).set_index("lag")
        self.lag_sensitivity.to_csv(self.out_dir / "lag_sensitivity.csv")

        print(self.lag_sensitivity.to_string())


    def _coint_and_vecm(self, prices, name, lag_diff):
        print(f"   {name}: {len(prices)} obs")

        # johansen test with constant term
        joh = coint_johansen(prices.values, det_order=0, k_ar_diff=lag_diff)

        # build results table with both trace and max-eigenvalue statistics
        results = pd.DataFrame({
            "trace_stat": joh.lr1,
            "trace_crit_5pct": joh.cvt[:, 1],
            "trace_reject": joh.lr1 > joh.cvt[:, 1],
            "max_eig_stat": joh.lr2,
            "max_eig_crit_5pct": joh.cvm[:, 1],
            "max_eig_reject": joh.lr2 > joh.cvm[:, 1],
        }, index=[f"r<={i}" for i in range(len(joh.lr1))])

        # cointegrating rank = number of rejections, testing sequential 
        rank_trace = int(results["trace_reject"].sum())
        rank_maxeig = int(results["max_eig_reject"].sum())

        print(f"   rank (trace)   = {rank_trace}")
        print(f"   rank (max-eig) = {rank_maxeig}")

        # use trace rank for VECM
        if rank_trace == 0:
            print(f"   no cointegration → skipping VECM")
            return results, None

        # fit VECM
        vecm = VECM(prices, k_ar_diff=lag_diff, coint_rank=rank_trace,
                    deterministic="ci").fit()

        # Top 3
        alpha_abs = pd.Series(
            abs(vecm.alpha[:, 0]), index=self.SECTORS
        ).sort_values(ascending=False)
        print(f"   leaders (top 3 by |α|): "
            f"{alpha_abs.index[0]}={alpha_abs.iloc[0]:.3f}, "
            f"{alpha_abs.index[1]}={alpha_abs.iloc[1]:.3f}, "
            f"{alpha_abs.index[2]}={alpha_abs.iloc[2]:.3f}")

        return results, vecm

    def _save_coint_results(self):
        self.coint_high.to_csv(self.out_dir / "johansen_high.csv")
        self.coint_low.to_csv(self.out_dir / "johansen_low.csv")

        for vecm, label in [(self.vecm_high, "high"), (self.vecm_low, "low")]:
            if vecm is None:
                continue
            cols = [f"vec_{i+1}" for i in range(vecm.alpha.shape[1])]
            # alpha: adjustment speeds, how fast it corrects
            alpha = pd.DataFrame(vecm.alpha, index=self.SECTORS, columns=cols)
            alpha.to_csv(self.out_dir / f"vecm_alpha_{label}.csv")
            # beta: cointegrating vector, equilibrium relationship
            beta = pd.DataFrame(vecm.beta, index=self.SECTORS, columns=cols)
            beta.to_csv(self.out_dir / f"vecm_beta_{label}.csv")



    def _plot_adjustment_speeds(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, vecm, title in [(axes[0], self.vecm_low, "Non-High-TPU"),
                                (axes[1], self.vecm_high, "High-TPU")]:
            if vecm is None:
                ax.text(0.5, 0.5, "no cointegration", ha="center", va="center")
                ax.set_title(f"adjustment speeds — {title}")
                ax.axis("off")
                continue

            cols = [f"vec_{i+1}" for i in range(vecm.alpha.shape[1])]
            alpha = pd.DataFrame(vecm.alpha, index=self.SECTORS, columns=cols)
            # sort by |alpha| descending so leaders appear at the top
            order = alpha[cols[0]].abs().sort_values(ascending=False).index
            alpha = alpha.reindex(order)

            sns.heatmap(alpha, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, ax=ax, cbar_kws={"label": "α"})
            ax.set_title(f"adjustment speeds α — {title}")

        plt.tight_layout()
        plt.savefig(self.out_dir / "vecm_alpha.png", dpi=150, bbox_inches="tight")
        plt.close()

    
    # ORCHESTRATOR 
    
    def run_dag_backdoor(self):
        """Function to run steps 1-3"""
        print("")
        print(" Data + DAG + Backdoor Adjustment")
        print("")

        self.load_data()
        self.build_dag()
        self.apply_backdoor_adjustment()

        print("")
        print("DAG + Backdoor Adjustment done")
        print("")


    def run_network_granger_fdr(self):
        print("Network + Granger + FDR + Metrics")
        self.compute_correlations()
        self.run_granger()
        self.apply_fdr()
        self.network_metrics()
        print("Networks and more done")


    def run_coin_vecm(self):
        print("Cointegration + VECM + Lag Sensitivity")
        self.run_johansen_vecm()
        self.lag_sensitivity_check()
        print("Cointegration and VECM done")

if __name__ == "__main__":
    ci = CausalInferenceAnalysis()
    ci.run_batch_a()