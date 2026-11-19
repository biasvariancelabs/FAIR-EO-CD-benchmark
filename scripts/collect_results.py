"""Collect the change detection runs from the MLflow export into one results file.

The export is a JSON list with one entry per MLflow run (metadata, parameters, tags, metrics and
artifacts). This script writes one row per run with the dataset, model, regime, every test metric,
the best epoch, the wall-clock time, the run id and the checkpoint path. All tables, figures and
statistics are generated from the file it writes.

Usage, from the root of the repository:
    python scripts/collect_results.py --export change_detection_runs.json --out results/runs.csv
"""
import argparse
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from results_common import DATASETS, MODELS, NO_PRETRAINED, REGIMES

FINISHED = 3  # MLflow RunStatus.FINISHED
KEY = ["dataset", "model", "regime"]


def metric_name(key):
    """'IOU mean/test' -> 'iou_mean', 'F1_score Macro/test' -> 'f1_score_macro'."""
    return re.sub(r"[^0-9a-z]+", "_", key.removesuffix("/test").lower()).strip("_")


def read_export(path, task):
    rows = []
    for entry in json.load(open(path, encoding="utf-8")):
        md, tags, params, metrics = entry["metadata"], entry["tags"], entry["parameters"], entry["metrics"]
        if tags.get("task") != task:
            continue
        pretrained = str(params.get("pretrained")).lower() == "true"
        if pretrained != ("_pretrained" in md["run_name"]):
            raise SystemExit(f"Run {md['run_id']}: the pretrained parameter does not match the run name {md['run_name']}")
        checkpoints = [a for a in entry.get("artifacts", []) if "/checkpoints/" in a]
        row = {
            "dataset": tags["dataset"],
            "model": tags["model_type"],
            "regime": "pretrained" if pretrained else "scratch",
            "run_id": md["run_id"],
            "run_name": md["run_name"],
            "status": md["status"],
            "start_utc": datetime.fromtimestamp(md["start_time"] / 1000, timezone.utc).isoformat(timespec="seconds"),
            "hours": (md["end_time"] - md["start_time"]) / 3.6e6 if md.get("end_time") else np.nan,
            # The best epoch is the one with the lowest validation loss. It is missing when the validation loss was NaN.
            "best_epoch": int(tags["best_epoch"]) if "best_epoch" in tags else np.nan,
            "gflops": metrics.get("GFLOPS_forward_pass", np.nan),
            "checkpoint": checkpoints[0] if checkpoints else "",
            "git_commit": tags.get("mlflow.source.git.commit", ""),
        }
        row.update({metric_name(k): v for k, v in metrics.items() if k.endswith("/test")})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="JSON export of the MLflow runs")
    parser.add_argument("--out", default="results/runs.csv")
    parser.add_argument("--task", default="change_detection")
    args = parser.parse_args()

    df = read_export(args.export, args.task)
    print(f"{len(df)} {args.task} runs in {args.export}")

    unknown = (set(df["model"]) - set(MODELS)) | (set(df["dataset"]) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown models or datasets: {sorted(unknown)}")
    unfinished = df[df["status"] != FINISHED]
    if len(unfinished):
        print(f"WARNING: {len(unfinished)} runs did not finish and are left out: {unfinished['run_name'].tolist()}")
        df = df[df["status"] == FINISHED]

    df = df.sort_values("start_utc")
    repeated = df[df.duplicated(KEY, keep=False)]
    if len(repeated):
        print(f"WARNING: several finished runs for {repeated[KEY].drop_duplicates().values.tolist()}, keeping the latest")
        df = df.drop_duplicates(KEY, keep="last")

    expected = {(d, m, r) for d in DATASETS for m in MODELS for r in REGIMES}
    missing = sorted(expected - set(map(tuple, df[KEY].values)))
    print(f"{len(df)} of {len(expected)} configurations have a finished run" + (f", missing: {missing}" if missing else ""))

    metric_cols = [c for c in df.columns if c not in KEY + ["run_id", "run_name", "status", "start_utc", "hours",
                                                             "best_epoch", "gflops", "checkpoint", "git_commit"]]
    incomplete = df[df[metric_cols].isna().any(axis=1) | ~np.isfinite(df[metric_cols]).all(axis=1)]
    if len(incomplete):
        print(f"WARNING: missing or non-finite test metrics in {incomplete['run_name'].tolist()}")
    nan_runs = df[df["best_epoch"].isna()]
    if len(nan_runs):
        print(f"WARNING: {len(nan_runs)} runs have no best epoch (NaN validation loss): "
              f"{nan_runs[KEY].values.tolist()}")
    print(f"The pretrained runs of {sorted(NO_PRETRAINED)} start from random weights. They are kept here "
          "but left out of the tables and statistics.")

    order = {"dataset": list(DATASETS), "model": list(MODELS), "regime": REGIMES}
    df = df.sort_values(KEY, key=lambda s: s.map({v: i for i, v in enumerate(order[s.name])}))
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(f"Wrote {args.out}: {len(df)} runs, metrics: {', '.join(metric_cols)}")


if __name__ == "__main__":
    main()
