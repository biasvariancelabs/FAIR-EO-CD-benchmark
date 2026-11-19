"""Efficiency summary from the output of profile_models.py.

Prints the activation memory, latency and throughput of each model in the row order of the
efficiency table in the paper, the Spearman rank correlations between parameters, FLOPs,
latency, throughput and memory, and the time to process one Sentinel-2 tile.

Usage, from the root of the repository:
    python scripts/efficiency.py --profiling results/profiling_results.csv
"""
import argparse
import math

import pandas as pd
from scipy import stats

# Model names as in the paper, in the row order of the efficiency table
NAMES = {"HRNetChangeDetection": "HRNet SiamConc", "UnetChangeDetection": "U-Net SiamConc"}
ORDER = ["BIT", "CGNet", "ChangeFormerV6", "ChangeViT", "CSSM", "HRNet SiamConc",
         "SiamCRNN", "STANet", "TinyCD", "U-Net SiamConc"]
TILE_PX = 10980  # one Sentinel-2 tile at 10 m resolution
PATCH_PX = 256  # input size of the models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiling", default="results/profiling_results.csv")
    args = parser.parse_args()

    p = pd.read_csv(args.profiling)
    failed = p[p["error"].notna()]
    if len(failed):
        raise SystemExit(f"Profiling failed for: {', '.join(failed['model'])}")
    p["name"] = p["model"].map(lambda m: NAMES.get(m, m))
    p = p.set_index("name").loc[ORDER]

    print("Efficiency table: activation memory (GB) & latency at batch size 1 (ms) & throughput at batch size 16 (pairs/s)")
    for name, r in p.iterrows():
        print(f"  {name} & {r.train_mem_activations_gb:.2f} & {r.latency_ms_median:.2f} & {r.throughput_pairs_per_s:.2f}")

    print("\nSpearman rank correlations over the models")
    for x, y in [("gflops", "latency_ms_median"), ("gflops", "throughput_pairs_per_s"),
                 ("params_total_m", "train_mem_activations_gb"), ("latency_ms_median", "throughput_pairs_per_s")]:
        rho, pvalue = stats.spearmanr(p[x], p[y])
        print(f"  {x} vs {y}: rho = {rho:+.2f} (p = {pvalue:.2f})")

    n = math.ceil(TILE_PX / PATCH_PX) ** 2
    print(f"\nTime for one Sentinel-2 tile ({TILE_PX} x {TILE_PX} px, {n} patches of {PATCH_PX} px, without data loading)")
    for name, r in p.sort_values("throughput_pairs_per_s", ascending=False).iterrows():
        print(f"  {name}: {n / r.throughput_pairs_per_s:.1f} s at batch size 16, "
              f"{n * r.latency_ms_median / 1000:.1f} s at batch size 1")


if __name__ == "__main__":
    main()
