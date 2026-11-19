"""Write the result tables of the paper and the README from results/runs.csv.

Writes, into the output folder:
  table_miou.tex        mIoU per dataset and model, from scratch and pretrained (Table 3)
  table_f1.tex          the same for the macro F1-score
  table_efficiency.tex  memory, latency, throughput and training time (Table 4), if --profiling is given
  readme_tables.md      the two mIoU tables of the README

In every row the best value is in bold and the second best is underlined, comparing the values
as printed (4 decimals), so that equal printed values get the same mark.

Usage, from the root of the repository:
    python scripts/make_tables.py --runs results/runs.csv --profiling results/profiling_results.csv --out results/tables
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from results_common import DATASETS, MODELS, NO_PRETRAINED, load_runs, matrix

MIOU, F1 = "iou_mean", "f1_score_macro"
CAPTIONS = {
    MIOU: (r"Mean Intersection over Union (mIoU) benchmark results across ten change detection datasets, comparing "
           r"models trained from scratch versus pre-trained backbones. The best performance per dataset within each "
           r"training setup is indicated in \textbf{bold}, and the second-best is \underline{underlined}.", "tab:miou"),
    F1: (r"Macro $F_1$-score benchmark results across ten change detection datasets, comparing models trained from "
         r"scratch versus pre-trained backbones. The best performance per dataset within each training setup is "
         r"indicated in \textbf{bold}, and the second-best is \underline{underlined}.", "tab:f1"),
}
EFFICIENCY_CAPTION = (
    r"Resource profiling and computational efficiency of the benchmarked models. Peak activation memory is estimated "
    r"for a training batch of size 16. Latency and throughput are measured on a single GPU with bi-temporal input "
    r"tensor of size $2 \times 3 \times 256 \times 256$ (batch size 1 for latency and 16 for throughput). Training "
    r"times represent the average wall-clock duration to reach convergence across all benchmark datasets.")
NA_NOTE = " %s means that no weights pre-trained on ImageNet-1K are available for the model."


def marks(values):
    """'best' and 'second' for the highest and second highest value as printed with 4 decimals."""
    printed = [None if np.isnan(v) else float(f"{v:.4f}") for v in values]
    distinct = sorted({p for p in printed if p is not None}, reverse=True)
    best, second = (distinct + [None, None])[:2]
    return [None if p is None else "best" if p == best else "second" if p == second else None for p in printed]


def cells(values, style, na):
    """Formatted values with bold for the best and underline (LaTeX) or <ins> (Markdown) for the second best."""
    wrap = {"latex": {"best": r"\textbf{%s}", "second": r"\underline{%s}"},
            "md": {"best": "**%s**", "second": "<ins>%s</ins>"}}[style]
    return [na if np.isnan(v) else (wrap[m] % f"{v:.4f}" if m else f"{v:.4f}") for v, m in zip(values, marks(values))]


def latex_results_table(df, metric, na):
    caption, label = CAPTIONS[metric]
    caption += NA_NOTE % na
    header = (r"    \textbf{Dataset} & " + " & ".join(r"\rotatebox{90}{\textbf{%s}}" % n for n in MODELS.values())
              + r" & \rotatebox{90}{\textbf{Average}} \\")
    lines = [r"\begin{table}[htbp]", r"    \centering", r"    \caption{%s}" % caption, r"    \vspace{5pt}",
             r"    \resizebox{\textwidth}{!}{", r"    \begin{tabular}{|l|cccccccccc|c|}", r"    \hline"]
    for i, (regime, title) in enumerate([("scratch", "(a) Trained from scratch"), ("pretrained", "(b) Pre-trained")]):
        if i:
            lines.append(r"    \hline")
        lines += [r"    \multicolumn{12}{|c|}{\textbf{%s}} \\" % title, r"    \hline", header, r"    \hline"]
        m = matrix(df, metric, regime)
        for ds, row in m.iterrows():
            avg = np.nanmean(row.values)
            lines.append(f"    {DATASETS[ds]} & " + " & ".join(cells(row.values, "latex", na)) + f" & {avg:.4f} \\\\")
        lines += [r"    \hline", r"    \textbf{Average} & " + " & ".join(cells(m.mean().values, "latex", na)) + r" & \\",
                  r"    \hline"]
    lines += [r"    \end{tabular}", r"    }", r"    \label{%s}" % label, r"\end{table}"]
    return "\n".join(lines) + "\n"


def latex_efficiency_table(df, profiling, na):
    prof = pd.read_csv(profiling).set_index("model")
    lines = [r"\begin{table}[!htbp]", r"  \centering", r"  \caption{%s}" % (EFFICIENCY_CAPTION + NA_NOTE % na),
             r"  \vspace{5pt}",
             r"  \small", r"  \resizebox{\textwidth}{!}{%", r"  \begin{tabular}{|l|c|c|c|c|c|}", r"    \hline",
             r"    \textbf{Model} & \textbf{Activation} & \textbf{Latency} & \textbf{Throughput} & "
             r"\multicolumn{2}{c|}{\textbf{Average train time (h)}} \\",
             r"    \cline{5-6}",
             r"    & \textbf{memory (GB)} & \textbf{(ms)} & \textbf{(pairs/s)} & \textbf{Scratch} & \textbf{Pre-trained} \\",
             r"    \hline"]
    for model, name in MODELS.items():
        p = prof.loc[model]
        times = []
        for regime in ["scratch", "pretrained"]:
            h = df[(df["model"] == model) & (df["regime"] == regime)]["hours"]
            # mean and sample standard deviation over the datasets
            times.append(na if h.empty else f"{h.mean():.2f}$\\pm${h.std():.2f}")
        lines += [f"    {name} & {p.train_mem_activations_gb:.2f} & {p.latency_ms_median:.2f} & "
                  f"{p.throughput_pairs_per_s:.2f} & {times[0]} & {times[1]} \\\\", r"    \hline"]
    lines += [r"  \end{tabular}%", r"  }", r"    \label{tab:compute_efficiency}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def readme_tables(df):
    out = []
    for regime, intro in [("scratch", "Mean Intersection over Union (mIoU) for models trained from scratch. The best "
                                      "performance is indicated in bold, and the second-best is underlined."),
                          ("pretrained", "Mean Intersection over Union (mIoU) for models with pre-trained backbones.")]:
        m = matrix(df, MIOU, regime).dropna(axis=1, how="all")
        names = [MODELS[c] for c in m.columns]
        out += [intro, "", "| Dataset/Model | " + " | ".join(names) + " | Average |", "|---" * (len(names) + 2) + "|"]
        for ds, row in m.iterrows():
            out.append(f"| **{DATASETS[ds]}** | " + " | ".join(cells(row.values, "md", "")) + f" | {row.mean():.4f} |")
        out += ["| **Average** | " + " | ".join(cells(m.mean().values, "md", "")) + " | |", "", ""]
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--profiling", help="profiling_results.csv from profile_models.py")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--na", default="N/A", help="cell text where a model has no pretrained weights")
    args = parser.parse_args()

    df = load_runs(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"table_miou.tex": latex_results_table(df, MIOU, args.na),
             "table_f1.tex": latex_results_table(df, F1, args.na),
             "readme_tables.md": readme_tables(df)}
    if args.profiling:
        files["table_efficiency.tex"] = latex_efficiency_table(df, args.profiling, args.na)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {out / name}")
    print(f"Left out: the pretrained runs of {sorted(NO_PRETRAINED)} (no pretrained weights)")


if __name__ == "__main__":
    main()
