"""Rank statistics across datasets from results/runs.csv.

For each regime: average ranks of the models over the datasets, the Friedman test and the Nemenyi
critical difference (alpha = 0.05). For each model with pretrained weights: the effect of
pretraining over the datasets with the Wilcoxon signed-rank test and Holm's correction. Finally,
Kendall's tau between the rankings of the two regimes. Each dataset counts as one observation
(Demsar, 2006).

Usage, from the root of the repository:
    python scripts/stats.py --runs results/runs.csv --metric iou_mean --out results/stats
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from results_common import MODELS, REGIMES, load_runs, matrix

# Critical values q_0.05 of the Nemenyi test (Demsar, 2006, Table 5a), by number of models
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def holm(p):
    """Holm-adjusted p-values, in the original order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--metric", default="iou_mean")
    parser.add_argument("--out", default="results/stats")
    args = parser.parse_args()

    df = load_runs(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ranks_table = []
    for regime in REGIMES:
        m = matrix(df, args.metric, regime).dropna(axis=1, how="all")
        if m.isna().any().any():
            raise SystemExit(f"{regime}: missing values for {m.columns[m.isna().any()].tolist()}")
        n, k = m.shape
        ranks = m.rank(axis=1, ascending=False)  # 1 = best, ties share the average rank
        chi2, p = stats.friedmanchisquare(*[m[c] for c in m.columns])
        cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
        print(f"\n{regime}: {k} models, {n} datasets, Friedman chi2 = {chi2:.2f}, p = {p:.2g}, Nemenyi CD = {cd:.2f}")
        summary = pd.DataFrame({"model": [MODELS[c] for c in m.columns], "regime": regime, "mean": m.mean().values,
                                "average_rank": ranks.mean().values, "friedman_chi2": chi2, "friedman_p": p,
                                "nemenyi_cd": cd}).sort_values("average_rank")
        print(summary[["model", "mean", "average_rank"]].round(4).to_string(index=False))
        ranks_table.append(summary)
    pd.concat(ranks_table).to_csv(out / f"{args.metric}_ranks.csv", index=False, lineterminator="\n")

    # Paired effect of pretraining for each model, over the datasets
    both = {r: matrix(df, args.metric, r) for r in REGIMES}
    rows = []
    for model in MODELS:
        s, p = both["scratch"][model], both["pretrained"][model]
        if p.isna().all():
            continue
        gain = p - s
        rows.append({"model": MODELS[model], "mean_gain": gain.mean(), "median_gain": gain.median(),
                     "datasets_improved": int((gain > 0).sum()), "datasets": int(gain.notna().sum()),
                     "wilcoxon_p": stats.wilcoxon(p, s).pvalue})
    effect = pd.DataFrame(rows)
    effect["holm_p"] = holm(effect["wilcoxon_p"])
    print("\nEffect of pretraining (pretrained minus scratch)")
    print(effect.round(4).to_string(index=False))
    effect.to_csv(out / f"{args.metric}_pretraining.csv", index=False, lineterminator="\n")

    keep = [m for m in MODELS if both["pretrained"][m].notna().any()]
    tau, p = stats.kendalltau(both["scratch"][keep].mean(), both["pretrained"][keep].mean())
    print(f"\nKendall tau between the average {args.metric} of the {len(keep)} models in both regimes: "
          f"{tau:.2f} (p = {p:.2g})")
    print(f"Wrote {out / (args.metric + '_ranks.csv')} and {out / (args.metric + '_pretraining.csv')}")


if __name__ == "__main__":
    main()
