"""Make the report figures from results/*.json (skips any not yet run).

Outputs PDF figures in report/figures/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import FIGURES, RESULTS

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def recall():
    r = load("recall_replication.json")
    if not r:
        return
    groups = [(d, c) for d in r["results"] for c in r["results"][d]]
    labels = [f"{d}\n{'sparse LR' if c == 'l1logreg' else 'tree'}" for d, c in groups]
    explainers = [e for e in ("random", "greedy", "lime") if e in r["results"][groups[0][0]][groups[0][1]]]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(groups))
    w = 0.8 / len(explainers)
    colors = {"random": AQUA, "greedy": ORANGE, "lime": BLUE}
    for k, e in enumerate(explainers):
        ours = [r["results"][d][c][e]["recall_pct"] for d, c in groups]
        paper = [r["paper"][d][c][e] for d, c in groups]
        xs = x - 0.4 + w * (k + 0.5)
        ax.bar(xs, ours, w * 0.9, color=colors[e], label=e.upper() if e == "lime" else e)
        ax.scatter(xs, paper, marker="_", s=60, color=INK, linewidths=1.4, zorder=3,
                   label="paper" if k == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Recall of gold features (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
    save(fig, "recall.pdf")


def constructed():
    data = {m: load(f"constructed_{m}.json") for m in ("binary", "tfidf")}
    if not all(data.values()):
        return
    conds = ["neg100", "neg75", "neg50", "neg25", "control"]
    labels = ["100", "75", "50", "25", "ctrl"]
    names = {"binary": "binary BoW", "tfidf": "TF-IDF"}
    cols = {"binary": BLUE, "tfidf": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    x = np.arange(len(conds))
    for ax, key, title in ((axes[0], "detect_at_1", "(a) Token ranked first by LIME"),
                           (axes[1], "detect_at_10", "(b) Token in LIME's top 10")):
        for k, m in enumerate(("binary", "tfidf")):
            s = data[m]["summary"]
            mean = [100 * s[c][key]["mean"] for c in conds]
            sd = [100 * s[c][key]["sd"] for c in conds]
            xs = x + (k - 0.5) * 0.4
            ax.bar(xs, mean, 0.36, yerr=sd, color=cols[m], label=names[m],
                   error_kw={"elinewidth": 0.7, "ecolor": MUTED, "capsize": 1.5})
            for xi, v in zip(xs, mean):
                if v < 3:
                    ax.text(xi, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=MUTED)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 112)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Negative reviews with token (%)")
    axes[0].set_ylabel("Injected test reviews (%)")
    ax = axes[2]
    for m in ("binary", "tfidf"):
        runs = [q for q in data[m]["runs"] if q["condition"] != "control"]
        xs_, ys_ = [q["mean_prob_drop"] for q in runs], [q["mean_abs_lime_weight"] for q in runs]
        r = np.corrcoef(xs_, ys_)[0, 1]
        ax.scatter(xs_, ys_, s=12, color=cols[m], edgecolors="white", linewidths=0.4,
                   label=f"{names[m]} (r = {r:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model reliance (drop in P(neg))")
    ax.set_ylabel("|LIME weight| on token")
    ax.set_title("(c) Reliance vs LIME weight, per run", fontsize=8, color=INK, loc="left")
    ax.legend(fontsize=6.5, loc="lower right", handletextpad=0.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.36, 1.06))
    save(fig, "constructed.pdf")


def tabular():
    h, ad = load("heart_disease.json"), load("adult_income.json")
    sets = [(n, d) for n, d in (("Heart Disease", h), ("Adult Income", ad)) if d]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(3.3 * len(sets), 2.3))
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, sets):
        f = d["summary"]["feature_freq_top5"][:8][::-1]
        y = np.arange(len(f))
        ax.barh(y, [100 * q["mean"] for q in f], xerr=[100 * q["sd"] for q in f], color=BLUE, height=0.6,
                error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capsize": 2})
        ax.set_yticks(y, [q["feature"] for q in f])
        ax.set_xlabel("Explanations with feature in top 5 (%)")
        ax.set_xlim(0, 105)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{name}: ρ vs permutation importance = {d['summary']['spearman_mean']:.2f}",
                     fontsize=8, color=INK, loc="left")
    save(fig, "tabular.pdf")


if __name__ == "__main__":
    recall()
    constructed()
    tabular()
