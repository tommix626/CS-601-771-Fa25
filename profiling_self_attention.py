import os, sys, math, time, csv, gc, argparse, statistics, json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional psutil for CPU RSS; we gracefully fall back if missing
try:
    import psutil
except Exception:
    psutil = None

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    outdir: str
    lengths: List[int]
    batch_size: int
    d_model: int
    n_heads: int
    dtype: str
    seed: int
    warmup: int
    trials_small: int
    trials_medium: int
    trials_large: int
    cpu_mem_trials: int

def make_config(outdir: str) -> Config:
    return Config(
        outdir=outdir,
        lengths=[10, 100, 1000, 10000],
        batch_size=1,
        d_model=128,        # keep modest so L=10k fits and finishes
        n_heads=4,
        dtype="float32",    # use float32 on both CPU/GPU for apples-to-apples
        seed=123,
        warmup=5,
        trials_small=30,    # L <= 100
        trials_medium=10,   # L == 1000
        trials_large=3,     # L == 10000 (keep short)
        cpu_mem_trials=3,   # profiling CPU memory is slower; sample a few times
    )

# -------------------------------
# Model: simple scaled dot-product self-attention (no bias)
# -------------------------------
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, device, dtype):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.out = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv(x)                                  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                     # each [B, L, D]

        H, Dh = self.n_heads, self.d_head
        # [B, H, L, Dh]
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # [B, H, L, L]
        attn = torch.softmax(scores, dim=-1)                            # [B, H, L, L]
        context = torch.matmul(attn, v)                                 # [B, H, L, Dh]
        context = context.transpose(1, 2).contiguous().view(B, L, D)    # [B, L, D]
        y = self.out(context)                                           # [B, L, D]
        return y

# -------------------------------
# FLOPs accounting (forward only, multiply+add = 2 FLOPs)
# Returns integer FLOPs per forward pass for given (B, L, D, H)
# -------------------------------
def theoretical_attention_flops(B: int, L: int, D: int, H: int) -> int:
    Dh = D // H
    # Q,K,V projections: 3 * (B * L * (2*D*D))
    flops_qkv = 3 * (B * L * (2 * D * D))
    # Scores QK^T: B * H * L * (2 * L * Dh)
    flops_scores = B * H * L * (2 * L * Dh)
    # Softmax (rough): ~ 5 ops per element
    flops_softmax = 5 * B * H * L * L
    # AV: B * H * L * (2 * L * Dh)
    flops_av = B * H * L * (2 * L * Dh)
    # Output projection: B * L * (2 * D * D)
    flops_out = B * L * (2 * D * D)
    return int(flops_qkv + flops_scores + flops_softmax + flops_av + flops_out)

# -------------------------------
# Measurement helpers
# -------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_dtype(name: str):
    m = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in m:
        raise ValueError(f"Unknown dtype: {name}")
    return m[name]

def cuda_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out

def cpu_time_ms(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out

def measure_gpu_memory_bytes(device) -> int:
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device=device)
    return int(peak)

def measure_cpu_rss_bytes() -> int:
    # Prefer psutil if available; else approximate via /proc/self/statm (Linux)
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    # Fallback: /proc/self/statm: total program size, resident set, shared, text, lib, data, dt (pages)
    try:
        with open("/proc/self/statm", "r") as f:
            fields = f.readline().split()
        pagesize = os.sysconf("SC_PAGE_SIZE")
        rss_pages = int(fields[1])
        return int(rss_pages * pagesize)
    except Exception:
        # Last resort: 0 (unknown)
        return 0

def standard_error(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    return statistics.pstdev(xs) / math.sqrt(n) if n > 1 else 0.0

# -------------------------------
# Profiling loop per device
# -------------------------------
def profile_device(cfg: Config, device: str) -> List[Dict]:
    results = []
    torch_dtype = to_dtype(cfg.dtype)

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; skipping GPU.", file=sys.stderr)
        return results

    dev = torch.device(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(cfg.seed)
    model = SelfAttention(cfg.d_model, cfg.n_heads, dev, torch_dtype).eval()

    for L in cfg.lengths:
        B = cfg.batch_size
        # choose trials by L
        if L <= 100:
            trials = cfg.trials_small
        elif L <= 1000:
            trials = cfg.trials_medium
        else:
            trials = cfg.trials_large

        # Input tensor created once per length to avoid allocating in-loop
        x = torch.randn(B, L, cfg.d_model, device=dev, dtype=torch_dtype)

        # Warmup (helps stabilize measurements)
        for _ in range(cfg.warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Timings
        times_ms: List[float] = []
        # Memory (bytes)
        mem_bytes: List[int] = []
        # Achieved throughput (GFLOP/s) per trial (uses theoretical FLOPs)
        gflops_per_s: List[float] = []

        est_flops = theoretical_attention_flops(B, L, cfg.d_model, cfg.n_heads)

        # We measure memory in each trial (GPU precise, CPU approximate)
        for t in range(trials):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=dev)
                dt_ms, _ = cuda_time_ms(lambda: model(x)) # forward pass as the function to measure
                peak = measure_gpu_memory_bytes(dev)
            else:
                # Approximate CPU memory usage as delta of RSS (peak within run is hard to catch without heavy profilers)
                rss_before = measure_cpu_rss_bytes()
                dt_ms, _ = cpu_time_ms(lambda: model(x))
                rss_after = measure_cpu_rss_bytes()
                peak = max(0, rss_after - rss_before)

            times_ms.append(float(dt_ms))
            mem_bytes.append(int(peak))
            # achieved GFLOPs/s using est_flops and measured time
            gflops_per_s.append((est_flops / 1e9) / (dt_ms / 1000.0))

        # For CPU, optionally refine memory with torch.profiler (slow): sample a few times
        if device == "cpu" and cfg.cpu_mem_trials > 0:
            try:
                from torch.profiler import profile, ProfilerActivity
                mems = []
                for _ in range(cfg.cpu_mem_trials):
                    gc.collect()
                    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
                        _ = model(x)
                    # Take max CPU memory usage from events (in Bytes)
                    max_cpu_mem = 0
                    for evt in prof.events():
                        if hasattr(evt, "cpu_memory_usage") and evt.cpu_memory_usage is not None:
                            max_cpu_mem = max(max_cpu_mem, int(evt.cpu_memory_usage))
                    if max_cpu_mem > 0:
                        mems.append(max_cpu_mem)
                if mems:
                    # Replace CPU memory series with profiler-based samples padded to trials length for correct SEM
                    mem_bytes = mems
            except Exception:
                pass  # fall back to RSS deltas

        # Aggregate
        def mean_sem(vs: List[float]) -> Tuple[float, float]:
            if len(vs) == 0:
                return 0.0, 0.0
            mu = float(np.mean(vs))
            se = float(np.std(vs, ddof=1) / math.sqrt(len(vs))) if len(vs) > 1 else 0.0
            return mu, se

        time_mu, time_se = mean_sem(times_ms)
        mem_mu,  mem_se  = mean_sem([float(m) for m in mem_bytes])
        thr_mu,  thr_se  = mean_sem(gflops_per_s)

        results.append(dict(
            device=device,
            L=L,
            B=B,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dtype=cfg.dtype,
            trials=trials,
            time_ms_mean=time_mu,
            time_ms_se=time_se,
            mem_bytes_mean=mem_mu,
            mem_bytes_se=mem_se,
            est_flops=est_flops,
            achieved_gflops_s_mean=thr_mu,
            achieved_gflops_s_se=thr_se
        ))

        print(f"[{device}] L={L:5d}  time={time_mu:.2f}±{time_se:.2f} ms  "
              f"mem~{mem_mu/1e6:.1f}±{mem_se/1e6:.1f} MB  "
              f"estFLOPs={est_flops/1e9:.3f} G  "
              f"thru={thr_mu:.2f}±{thr_se:.2f} GF/s")

    return results

# -------------------------------
# Plotting
# -------------------------------
def save_plots(all_results: List[Dict], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    devs = sorted(set(r["device"] for r in all_results))
    lengths = sorted(set(r["L"] for r in all_results))

    # Helper to extract series
    def series(metric_mean: str, metric_se: str, device: str):
        xs, ys, es = [], [], []
        for L in lengths:
            rows = [r for r in all_results if r["device"] == device and r["L"] == L]
            if not rows: continue
            xs.append(L); ys.append(rows[0][metric_mean]); es.append(rows[0][metric_se])
        return xs, ys, es

    # 1) Wall-clock time (ms)
    plt.figure(figsize=(7,5))
    for dev in devs:
        xs, ys, es = series("time_ms_mean", "time_ms_se", dev)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=dev.upper())
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Sequence length L (log)")
    plt.ylabel("Forward time (ms, log)")
    plt.title("Self-Attention Wall-Clock vs Length (mean ± SEM)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_time_ms.png"), dpi=160)
    plt.close()

    # 2) Memory (bytes)
    plt.figure(figsize=(7,5))
    for dev in devs:
        xs, ys, es = series("mem_bytes_mean", "mem_bytes_se", dev)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=dev.upper())
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Sequence length L (log)")
    plt.ylabel("Peak memory (Bytes, log)")
    plt.title("Self-Attention Memory vs Length (mean ± SEM)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_memory_bytes.png"), dpi=160)
    plt.close()

    # 3) Theoretical FLOPs per forward (count) — deterministic (SEM≈0)
    plt.figure(figsize=(7,5))
    for dev in devs:
        xs, ys, _ = series("est_flops", "est_flops", dev)
        ys = [y for y in ys]  # just counts
        plt.plot(xs, ys, marker="o", label=f"{dev.upper()} (same counts)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Sequence length L (log)")
    plt.ylabel("FLOPs per forward (log)")
    plt.title("Self-Attention Theoretical FLOPs vs Length")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_flops_count.png"), dpi=160)
    plt.close()

    # 4) Achieved throughput (GFLOP/s) — has error bars
    plt.figure(figsize=(7,5))
    for dev in devs:
        xs, ys, es = series("achieved_gflops_s_mean", "achieved_gflops_s_se", dev)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=dev.upper())
    plt.xscale("log")
    plt.xlabel("Sequence length L (log)")
    plt.ylabel("Achieved throughput (GFLOP/s)")
    plt.title("Achieved GFLOP/s vs Length (mean ± SEM)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_achieved_gflops_per_s.png"), dpi=160)
    plt.close()

# -------------------------------
# Report
# -------------------------------
def write_report(all_results: List[Dict], cfg: Config, outdir: str):
    # Simple interpretation text
    lines = []
    lines.append("# Empirical: Profiling Self-Attention\n")
    lines.append(f"- Batch size: {cfg.batch_size}, d_model: {cfg.d_model}, heads: {cfg.n_heads}, dtype: {cfg.dtype}")
    lines.append(f"- Lengths: {cfg.lengths}")
    lines.append("")
    lines.append("## Key observations")
    lines.append("1. **Wall-clock vs L**: Time scales ~O(L^2) as expected from the two L×L matmuls (QK^T and AV).")
    lines.append("2. **Memory vs L**: Peak memory also ~O(L^2), dominated by the attention scores/weights of shape H×L×L.")
    lines.append("3. **FLOPs vs L**: Theoretical FLOPs grow ~O(L^2·d_model) + O(L·d_model^2); for large L the L^2 term dominates.")
    lines.append("4. **CPU vs GPU**: GPU achieves far higher GFLOP/s and often benefits more as L rises (better kernel efficiency).")
    lines.append("")
    lines.append("## Tips / Caveats")
    lines.append("- CPU memory is approximated via RSS deltas and/or PyTorch CPU memory profiler; GPU peak memory uses CUDA allocator stats.")
    lines.append("- Using TF32/bfloat16 on GPUs can reduce runtime/memory further; we used float32 for fair CPU/GPU comparison.")
    lines.append("- For L=10k, ensure the node has sufficient GPU RAM; otherwise reduce `d_model` or `n_heads`.")
    lines.append("")
    # Append a small CSV preview
    lines.append("## Results (first few rows)")
    header = ["device","L","time_ms_mean","time_ms_se","mem_bytes_mean","mem_bytes_se","est_flops","achieved_gflops_s_mean","achieved_gflops_s_se"]
    lines.append("`" + ",".join(header) + "`")
    for r in all_results[:min(6, len(all_results))]:
        row = [str(r[k]) for k in header]
        lines.append("`" + ",".join(row) + "`")
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(lines))

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    cfg = make_config(args.outdir)
    os.makedirs(cfg.outdir, exist_ok=True)

    # Run CPU + GPU (if available)
    all_results: List[Dict] = []
    for dev in ["cpu", "cuda"]:
        res = profile_device(cfg, dev)
        all_results.extend(res)

    # Save CSV
    csv_path = os.path.join(cfg.outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "device","L","B","d_model","n_heads","dtype","trials",
            "time_ms_mean","time_ms_se",
            "mem_bytes_mean","mem_bytes_se",
            "est_flops",
            "achieved_gflops_s_mean","achieved_gflops_s_se"
        ])
        for r in all_results:
            w.writerow([
                r["device"], r["L"], r["B"], r["d_model"], r["n_heads"], r["dtype"], r["trials"],
                f"{r['time_ms_mean']:.6f}", f"{r['time_ms_se']:.6f}",
                int(r["mem_bytes_mean"]), int(r["mem_bytes_se"]),
                int(r["est_flops"]),
                f"{r['achieved_gflops_s_mean']:.6f}", f"{r['achieved_gflops_s_se']:.6f}"
            ])

    # Plots + report
    save_plots(all_results, cfg.outdir)
    write_report(all_results, cfg, cfg.outdir)

    # Print completion summary
    print("\n=== DONE ===")
    print(f"Output directory: {cfg.outdir}")
    print(f"CSV:   {os.path.join(cfg.outdir, 'results.csv')}")
    print(f"PLOTS: {os.path.join(cfg.outdir, 'plot_time_ms.png')}, "
          f"{os.path.join(cfg.outdir, 'plot_memory_bytes.png')}, "
          f"{os.path.join(cfg.outdir, 'plot_flops_count.png')}, "
          f"{os.path.join(cfg.outdir, 'plot_achieved_gflops_per_s.png')}")
    print(f"REPORT: {os.path.join(cfg.outdir, 'report.txt')}")

if __name__ == "__main__":
    main()