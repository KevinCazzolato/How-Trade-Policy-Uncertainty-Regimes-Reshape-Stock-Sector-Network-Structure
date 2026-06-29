"""
Phase 5 - Tariff Event Study
============================================================================
Goal: show whether the 2025 tariff events actually changed the way the stock
market sectors move together, instead of just claiming they did.

The idea is simple. Around each tariff date we take a window of trading days
just BEFORE the event and just AFTER it, and we build a network of the 11
sectors from the real returns in each window (two sectors are linked when they
move together strongly). Then we ask three questions:

  1. Did the network get denser / more tangled after the event?
  2. Is that change bigger than what we'd see just by chance? (permutation test)
  3. Does the "after" network look like the High-TPU stress regime that Angel
     already found in Phase 3?

Everything here is computed from the data. Nothing about the networks is
hand-drawn or assumed in advance.
"""

import os
import warnings
import itertools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Settings. If a file path or a date changes, this is the only place to edit.
# ----------------------------------------------------------------------------
CONFIG = {
    "data_dir": "../data",
    "sectors": ["XLK", "XLI", "XLY", "XLF", "XLV", "XLE",
                "XLB", "XLRE", "XLU", "XLP", "XLC"],

    # daily sector returns (also has the regime label column) - from Phase 1/3
    "returns_file":     ("phase_3", "purged_returns.csv"),
    # Phase 3 regime correlation matrices, used only to cross-check our result
    "phase3_corr_high": ("phase_3", "corr_high.csv"),
    "phase3_corr_low":  ("phase_3", "corr_low.csv"),
    # where the plot and summary table get written
    "output_dir":       ("phase_5", ""),

    "date_col": "date",

    # The real 2025 tariff dates (China / Canada / Mexico). Each date is the
    # calendar day of the announcement; the code automatically uses the first
    # actual trading day on or after it for the "after" window.
    "events": {
        "Feb 1 2025: CA/MX/CN tariffs announced": "2025-02-01",
        "Mar 4 2025: tariffs take effect":        "2025-03-04",
        "Apr 2 2025: Liberation Day":             "2025-04-02",
    },
    "main_event": "Apr 2 2025: Liberation Day",   # the event we draw in detail

    "window": 30,            # how many trading days to take on each side
    "corr_threshold": 0.7,   # draw a link only if |correlation| is above this.
                             # NOTE: this only affects the picture and the
                             # density number. The two main numbers below
                             # (avg correlation and the Frobenius distance) do
                             # NOT depend on it, so the headline result is not
                             # sensitive to where we set this line.
    "n_permutations": 2000,  # how many random reshuffles for the chance test
    "seed": 42,
}


class TariffEventStudy:
    """Runs the whole Phase 5 analysis: load data, compare the network before
    and after each tariff event, test if the change is real, and draw it."""

    def __init__(self, config=CONFIG):
        self.cfg = config
        self.rng = np.random.default_rng(config["seed"])
        self.returns = None        # daily sector returns
        self.regime = None         # regime label per day (from Phase 2)
        self.corr_high_p3 = None   # Phase 3 High-TPU correlation matrix
        self.corr_low_p3 = None    # Phase 3 Low-TPU correlation matrix
        out = config["output_dir"]
        self.output_dir = os.path.join(config["data_dir"], out[0], out[1]) if out[1] \
            else os.path.join(config["data_dir"], out[0])
        os.makedirs(self.output_dir, exist_ok=True)

    def _path(self, t):
        """Small helper to turn a (subfolder, filename) pair into a full path."""
        sub, name = t
        return os.path.join(self.cfg["data_dir"], sub, name) if sub else \
            os.path.join(self.cfg["data_dir"], name)

    # ---- loading -----------------------------------------------------------
    def load(self):
        """Read the returns, the regime labels, and the Phase 3 regime
        correlation matrices we use later as a sanity check."""
        cfg = self.cfg
        dcol = cfg["date_col"]
        df = pd.read_csv(self._path(cfg["returns_file"]))
        if dcol not in df.columns:                     # handle an unnamed date column
            df = df.rename(columns={df.columns[0]: dcol})
        df[dcol] = pd.to_datetime(df[dcol])
        df = df.sort_values(dcol).set_index(dcol)
        if "regime_label" in df.columns:
            self.regime = df["regime_label"]           # keep it to label each window later
        self.returns = df[[s for s in cfg["sectors"] if s in df.columns]]

        # Phase 3 gave us one correlation matrix per regime. We reload them so
        # we can later ask "does our after-event network look like the stress
        # regime?". If they are missing we just skip that cross-check.
        def _load_corr(t):
            c = pd.read_csv(self._path(t), index_col=0)
            return c.reindex(index=cfg["sectors"], columns=cfg["sectors"])
        try:
            self.corr_high_p3 = _load_corr(cfg["phase3_corr_high"])
            self.corr_low_p3 = _load_corr(cfg["phase3_corr_low"])
            print("  [OK] Phase 3 regime correlation matrices loaded for cross-check.")
        except FileNotFoundError:
            print("  [WARN] Phase 3 corr matrices not found; skipping regime cross-check.")
        print(f"  Returns: {self.returns.shape[0]} days, "
              f"{self.returns.index.min().date()} -> {self.returns.index.max().date()}")

    # ---- window helpers ----------------------------------------------------
    def _windows(self, event_date):
        """Cut out the trading days just before and just after an event.
        'before' = the last `window` days strictly before the date,
        'after'  = the first `window` days on or after the date."""
        w = self.cfg["window"]
        ev = pd.Timestamp(event_date)
        idx = self.returns.index
        pre = self.returns[idx < ev].tail(w)
        post = self.returns[idx >= ev].head(w)
        return pre, post

    @staticmethod
    def _upper(m):
        """Take the upper triangle of a matrix as a flat list. A correlation
        matrix is symmetric, so this keeps each sector pair exactly once and
        drops the meaningless diagonal of 1.0s."""
        a = m.values if isinstance(m, pd.DataFrame) else m
        iu = np.triu_indices_from(a, k=1)
        return a[iu]

    def _metrics(self, corr):
        """Turn a correlation matrix into a network and read off two things:
        how dense the network is, and how central each sector is (PageRank).
        A link is added between two sectors only if they are correlated more
        strongly than the threshold."""
        thr = self.cfg["corr_threshold"]
        sectors = list(corr.columns)
        G = nx.Graph()
        G.add_nodes_from(sectors)
        for a, b in itertools.combinations(sectors, 2):
            c = corr.loc[a, b]
            if pd.notna(c) and abs(c) > thr:
                G.add_edge(a, b, weight=abs(c))
        density = nx.density(G)                         # 0 = no links, 1 = everything linked
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else \
            {s: 1 / len(sectors) for s in sectors}      # flat ranking if no links at all
        deg = dict(G.degree())
        return G, density, pr, deg

    # ---- permutation test --------------------------------------------------
    def _perm_test(self, pre, post):
        """Test whether the before/after change is real or could just be noise.

        We measure the size of the change as the distance between the two
        correlation matrices (Frobenius norm). Then we pretend there was no
        event: we pool all the days together, reshuffle them randomly into a
        fake 'before' and 'after' many times, and measure the change each time.
        If the real change is bigger than almost all the random ones, the event
        really did move the structure. The p-value is the share of random
        shuffles that produced a change as big as the real one."""
        obs = np.linalg.norm(self._upper(post.corr()) - self._upper(pre.corr()))
        pooled = pd.concat([pre, post])
        n_pre = len(pre)
        null = np.empty(self.cfg["n_permutations"])
        vals = pooled.values
        for i in range(self.cfg["n_permutations"]):
            perm = self.rng.permutation(len(pooled))
            a = pd.DataFrame(vals[perm[:n_pre]], columns=pooled.columns)
            b = pd.DataFrame(vals[perm[n_pre:]], columns=pooled.columns)
            null[i] = np.linalg.norm(self._upper(b.corr()) - self._upper(a.corr()))
        p = (np.sum(null >= obs) + 1) / (self.cfg["n_permutations"] + 1)  # +1 keeps p > 0
        return obs, p

    # ---- per-event analysis ------------------------------------------------
    def run_event(self, name, date):
        """Do the full before/after comparison for a single tariff event."""
        pre, post = self._windows(date)
        if len(pre) < 10 or len(post) < 10:             # not enough days to trust the result
            print(f"  [WARN] '{name}': not enough data in window, skipping.")
            return None
        cpre, cpost = pre.corr(), post.corr()
        _, d_pre, pr_pre, _ = self._metrics(cpre)
        _, d_post, pr_post, _ = self._metrics(cpost)
        # average strength of co-movement, with no threshold involved
        avg_pre = np.nanmean(np.abs(self._upper(cpre)))
        avg_post = np.nanmean(np.abs(self._upper(cpost)))
        frob, pval = self._perm_test(pre, post)

        # which sectors became the most central after the event
        delta_pr = {s: pr_post[s] - pr_pre[s] for s in pr_pre}
        gained = sorted(delta_pr, key=delta_pr.get, reverse=True)[:3]

        # which regime the after-window mostly falls in (from Phase 2 labels)
        regime_note = ""
        if self.regime is not None:
            post_reg = self.regime.reindex(post.index).mode()
            if len(post_reg):
                regime_note = str(post_reg.iloc[0])

        res = dict(name=name, date=date, n_pre=len(pre), n_post=len(post),
                   density_pre=d_pre, density_post=d_post,
                   avgcorr_pre=avg_pre, avgcorr_post=avg_post,
                   frobenius=frob, p_value=pval, top_gainers=gained,
                   post_regime=regime_note, cpre=cpre, cpost=cpost,
                   pr_pre=pr_pre, pr_post=pr_post)
        return res

    # ---- cross-check vs Phase 3 regime networks ----------------------------
    def _regime_similarity(self, cpost):
        """Compare our after-event network to Angel's two Phase 3 regime
        networks. We just correlate the two sets of sector-pair values: a high
        number means our after-event structure looks like that regime. We
        expect it to look more like the High-TPU (stress) regime."""
        if self.corr_high_p3 is None:
            return None
        v = self._upper(cpost)
        sim_high = np.corrcoef(v, self._upper(self.corr_high_p3))[0, 1]
        sim_low = np.corrcoef(v, self._upper(self.corr_low_p3))[0, 1]
        return sim_high, sim_low

    # ---- comparative plot for the main event -------------------------------
    def plot_main_event(self, res):
        """Draw the before and after networks side by side for one event.
        Node size = how central the sector is, gold = the two biggest hubs.
        The densities printed in the titles are the real computed numbers."""
        cfg = self.cfg
        Gpre, d_pre, pr_pre, _ = self._metrics(res["cpre"])
        Gpost, d_post, pr_post, _ = self._metrics(res["cpost"])
        pos = nx.circular_layout(Gpre)                  # same layout both sides for fair comparison

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Sector Correlation Network Rewiring around\n{res['name']}"
                     f"  (window = {cfg['window']} trading days, |r| > {cfg['corr_threshold']})",
                     fontsize=14, fontweight="bold", y=1.04)

        for ax, G, dens, pr, label in [
            (axes[0], Gpre, d_pre, pr_pre, "PRE"),
            (axes[1], Gpost, d_post, pr_post, "POST"),
        ]:
            hubs = sorted(pr, key=pr.get, reverse=True)[:2]   # two most central sectors
            colors = ["gold" if n in hubs else "lightsteelblue" for n in G.nodes()]
            sizes = [600 + 18000 * pr[n] for n in G.nodes()]  # bigger node = more central
            ax.set_title(f"{label}-event  (density = {dens:.3f}, "
                         f"avg|r| = {np.nanmean(np.abs(self._upper(res['cpre' if label=='PRE' else 'cpost']))):.3f})",
                         fontsize=12)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                                   node_size=sizes, edgecolors="black")
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.4)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
            ax.axis("off")

        save = os.path.join(self.output_dir, "event_network_rewiring.png")
        plt.tight_layout()
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"  Plot saved: {save}")
        return save

    # ---- orchestrator ------------------------------------------------------
    def run(self):
        """Run the analysis for every event, print a summary table, save it,
        and draw the main event."""
        print("\n--- LOADING DATA ---")
        self.load()
        print("\n--- EVENT STUDY: NETWORK REWIRING AROUND 2025 TARIFF EVENTS ---")
        rows = []
        results = {}
        for name, date in self.cfg["events"].items():
            r = self.run_event(name, date)
            if r is None:
                continue
            results[name] = r
            sim = self._regime_similarity(r["cpost"])
            r["sim_high"], r["sim_low"] = (sim if sim else (np.nan, np.nan))
            rows.append({
                "event": name,
                "post_regime": r["post_regime"],
                "density_pre": round(r["density_pre"], 3),
                "density_post": round(r["density_post"], 3),
                "avg|r|_pre": round(r["avgcorr_pre"], 3),
                "avg|r|_post": round(r["avgcorr_post"], 3),
                "frobenius": round(r["frobenius"], 3),
                "perm_p": round(r["p_value"], 4),
                "sim_to_High": round(r["sim_high"], 3),
                "sim_to_Low": round(r["sim_low"], 3),
                "top_centrality_gainers": ", ".join(r["top_gainers"]),
            })
        summary = pd.DataFrame(rows)
        pd.set_option("display.width", 200, "display.max_columns", 30)
        print("\n", summary.to_string(index=False))

        out_csv = os.path.join(self.output_dir, "event_study_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"\n  Summary saved: {out_csv}")

        if self.cfg["main_event"] in results:
            print(f"\n--- COMPARATIVE PLOT: {self.cfg['main_event']} ---")
            self.plot_main_event(results[self.cfg["main_event"]])

        # How to read the table:
        #   perm_p small        -> the network really did change around the event
        #   density/avg|r| up   -> sectors started moving together more (stress)
        #   sim_to_High > Low   -> the after-event network looks like the
        #                          High-TPU stress regime Angel found in Phase 3
        print("\nInterpretation guide: a low perm_p means the correlation structure "
              "changed more than chance around the event; sim_to_High vs sim_to_Low "
              "shows whether the post-event network resembles the Phase 3 High-TPU "
              "(stress) regime rather than the calm regime.")
        return summary


if __name__ == "__main__":
    TariffEventStudy().run()
