"""Profile the benchmark models: parameters, FLOPs, inference latency and throughput, training step time and memory.

All models get the same synthetic input: two images of 3 x 256 x 256 (the input size of the benchmark)
and, for the training step, a binary change mask with two channels (no change, change). Pretrained
weights are not loaded, because they do not change the architecture.

One row per model is written to the output CSV:
- params_total_m, params_trainable_m: number of parameters in millions.
- gflops: fvcore FlopCountAnalysis of one forward pass at batch size 1, model in eval mode. fvcore counts
  one fused multiply-add as one FLOP, so the value equals GMACs. Element-wise operations are not counted,
  and operators that fvcore does not support are listed in fvcore_unsupported_ops.
- latency_ms_*: forward pass at batch size 1, eval mode, torch.inference_mode and float32, as in
  evaluation. The inputs are already on the GPU, so the host-to-device transfer is not included.
  Every iteration is timed with CUDA events after a warm-up.
- throughput_pairs_per_s: image pairs per second for the same forward pass at the training batch size.
- train_step_ms_*: one training step at the training batch size (forward pass, loss, backward pass and
  Adam update) with bfloat16 autocast, as in training.
- train_mem_*_gb: GPU memory during these training steps, measured with torch.cuda.memory_allocated and
  torch.cuda.max_memory_allocated after warm-up steps, so that the Adam state exists.
  static: weights, Adam state and the input batch before a step.
  peak: maximum allocated memory during a step.
  activations: peak minus static, that is activations, gradients and temporary buffers.
  peak_reserved: maximum memory held by the PyTorch caching allocator.
  GB means 10^9 bytes.

The software and hardware environment is written to a JSON file next to the CSV.

Usage, from the repository root (the GPU index is an example):
    CUDA_VISIBLE_DEVICES=1 python scripts/profile_models.py --output profiling_results.csv
"""

import argparse
import contextlib
import csv
import gc
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from training_run import get_repo_commit, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
GB = 1e9

FIELDS = [
    "model",
    "params_total_m",
    "params_trainable_m",
    "gflops",
    "fvcore_unsupported_ops",
    "latency_ms_median",
    "latency_ms_mean",
    "latency_ms_std",
    "latency_ms_p95",
    "throughput_pairs_per_s",
    "train_step_ms_median",
    "train_step_ms_mean",
    "train_mem_static_gb",
    "train_mem_peak_gb",
    "train_mem_activations_gb",
    "train_mem_peak_reserved_gb",
    "train_precision",
    "error",
]


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def read_proc_value(path, key):
    """First value of a "key: value" line in a /proc file (Linux only)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def environment(device, model_conf, args):
    props = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    mem_kb = read_proc_value("/proc/meminfo", "MemTotal")
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": props.name,
        "gpu_memory_gb": round(props.total_memory / GB, 1),
        "nvidia_driver": driver,
        "cpu": read_proc_value("/proc/cpuinfo", "model name") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(int(mem_kb.split()[0]) * 1024 / GB, 1) if mem_kb else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "aitlas": package_version("aitlas"),
        "fvcore": package_version("fvcore"),
        "segmentation_models_pytorch": package_version("segmentation-models-pytorch"),
        "timm": package_version("timm"),
        "benchmark_commit": get_repo_commit(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "model_config": model_conf,
        "arguments": vars(args),
    }


def time_cuda(fn, iters, warmup):
    """Run fn warmup times, then time each of iters runs with CUDA events. Returns milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return np.array(times)


def profile_model(model_class, model_conf, args, device):
    row = {"model": model_class}
    shape = (3, args.img_size, args.img_size)

    set_seed(args.seed)
    model_cls = getattr(importlib.import_module("aitlas.models"), model_class)
    model = model_cls({
        "learning_rate": args.learning_rate,
        "pretrained": False,
        "automatic_mixed_precision": True,
        **model_conf,
    })
    model.prepare()  # creates the loss, the Adam optimizer and the scheduler, as in training
    net = model.model.to(device)
    criterion = model.criterion.to(device)
    optimizer = model.optimizer

    row["params_total_m"] = sum(p.numel() for p in net.parameters()) / 1e6
    row["params_trainable_m"] = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    # FLOPs of one forward pass at batch size 1
    x1 = torch.rand(1, *shape, device=device)
    x2 = torch.rand(1, *shape, device=device)
    net.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(net, (x1, x2))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        row["gflops"] = flops.total() / 1e9
        row["fvcore_unsupported_ops"] = ";".join(
            f"{op}:{count}" for op, count in sorted(flops.unsupported_ops().items())
        )

    # Inference in float32 at batch size 1 (latency) and at the training batch size (throughput)
    xb1 = torch.rand(args.batch_size, *shape, device=device)
    xb2 = torch.rand(args.batch_size, *shape, device=device)
    with torch.inference_mode():
        latency = time_cuda(lambda: net(x1, x2), args.latency_iters, args.latency_warmup)
        batch_times = time_cuda(lambda: net(xb1, xb2), args.throughput_iters, args.throughput_warmup)
    row["latency_ms_median"] = float(np.median(latency))
    row["latency_ms_mean"] = float(latency.mean())
    row["latency_ms_std"] = float(latency.std())
    row["latency_ms_p95"] = float(np.percentile(latency, 95))
    row["throughput_pairs_per_s"] = args.batch_size * 1000 / float(np.median(batch_times))

    # Training steps at the training batch size, with the precision AiTLAS uses for training
    change = (torch.rand(args.batch_size, 1, args.img_size, args.img_size, device=device) > 0.9).float()
    labels = torch.cat([1 - change, change], dim=1)
    amp_dtype = model.amp_dtype
    row["train_precision"] = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "float32"

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
        with autocast:
            outputs = net(xb1, xb2)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.train()
    for _ in range(args.train_warmup):
        train_step()
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    static = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    step_times = time_cuda(train_step, args.train_iters, warmup=0)
    peak = torch.cuda.max_memory_allocated(device)
    row["train_step_ms_median"] = float(np.median(step_times))
    row["train_step_ms_mean"] = float(step_times.mean())
    row["train_mem_static_gb"] = static / GB
    row["train_mem_peak_gb"] = peak / GB
    row["train_mem_activations_gb"] = (peak - static) / GB
    row["train_mem_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / GB
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="model classes to profile (default: all in configs/models.json)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="profiling_results.csv")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--latency_warmup", type=int, default=50)
    parser.add_argument("--latency_iters", type=int, default=200)
    parser.add_argument("--throughput_warmup", type=int, default=10)
    parser.add_argument("--throughput_iters", type=int, default=50)
    parser.add_argument("--train_warmup", type=int, default=3)
    parser.add_argument("--train_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available, profiling needs a GPU.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Importing aitlas.models seeds torch with 1234, so the seed is set again for every model
    importlib.import_module("aitlas.models")

    with open(REPO_ROOT / "configs" / "models.json") as f:
        all_models = list(json.load(f)["change_detection"])
    with open(REPO_ROOT / "configs" / "datasets.json") as f:
        # The model configuration is the same for all datasets
        model_conf = next(iter(json.load(f)["change_detection"].values()))["model_config"]
    models = args.models or all_models

    output = Path(args.output)
    env_file = output.with_name(output.stem + "_env.json")
    env = environment(device, model_conf, args)
    with open(env_file, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Profiling {len(models)} models on {device} ({env['gpu']}), environment written to {env_file}")

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for model_class in models:
            print(f"--- {model_class}", flush=True)
            try:
                row = profile_model(model_class, model_conf, args, device)
            except Exception as e:  # keep going, so that one failing model does not stop the others
                traceback.print_exc()
                row = {"model": model_class, "error": repr(e)}
            row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            print(row, flush=True)
            writer.writerow(row)
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
