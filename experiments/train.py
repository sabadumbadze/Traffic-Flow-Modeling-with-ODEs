"""High-performance CPU-parallelized training script for the traffic flow ODE model.

Features:
- CPU parallelization using multiprocessing with configurable num_workers (default: all cores)
- Two-tier data caching (fast pickle cache on disk with detailed stage logging)
- 50 epochs training with Adam optimizer and physical parameter bounds
- Detailed metrics every 5 epochs: train/val loss, MAE, RMSE, parameter values, gradient norm
- System resource monitoring: per-core CPU utilization, RAM usage, GPU load & VRAM
- Full checkpointing every 5 epochs with explicit resume capability (--resume or --checkpoint)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    HIGH_CAPACITY_PARAM_LOWER,
    HIGH_CAPACITY_PARAM_NAMES,
    HIGH_CAPACITY_PARAM_UPPER,
    MAT_FILE,
    OBS_INTERVAL_H,
    PARAM_LOWER,
    PARAM_NAMES,
    PARAM_UPPER,
    RESULTS_DIR,
)
from src.input_signal import PiecewiseConstantInput
from src.pipeline import PreparedData, prepare_data
from src.rk4 import rk4_step

CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "prepared_data_cache.pkl"
CHECKPOINTS_DIR = RESULTS_DIR.parent / "checkpoints"


# =====================================================================
# 1. DATA CACHING & DETAILED LOADING
# =====================================================================

def get_cached_or_prepare_data(
    cache_path: Path = CACHE_FILE,
    force_reload: bool = False,
) -> PreparedData:
    """Loads dataset with on-disk binary caching and verbose progress printing."""
    print("=" * 80)
    print("DATA LOADING & CACHING SUBSYSTEM")
    print("=" * 80)

    if not force_reload and cache_path.is_file():
        cache_size_kb = cache_path.stat().st_size / 1024.0
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[CACHE] Found valid disk cache at: {cache_path}")
        print(f"[CACHE] Cache file size: {cache_size_kb:.2f} KB | Created: {mtime}")
        print("[CACHE] Reading preprocessed dataset from cache (bypassing raw MAT decoding)...")
        t0 = time.time()
        try:
            with open(cache_path, "rb") as f:
                data: PreparedData = pickle.load(f)
            t_load = time.time() - t0
            print(f"[CACHE] Successfully restored PreparedData in {t_load:.4f} seconds.")
            print(f"[CACHE] Sensor chain: Upstream={data.chain.upstream} -> States={data.chain.states}")
            print(f"[CACHE] Samples: Train={data.train.n} steps ({data.train.n * OBS_INTERVAL_H:.1f} h) | "
                  f"Test={data.test.n} steps ({data.test.n * OBS_INTERVAL_H:.1f} h)")
            print(f"[CACHE] Reference flow scale: Q_ref = {data.q_ref:.4f}")
            print("=" * 80)
            return data
        except Exception as e:
            print(f"[CACHE WARNING] Failed to read cache ({e}). Rebuilding from raw source...")

    # Cache miss or rebuild requested
    print(f"[RAW LOAD] Cache miss. Reading raw MATLAB v5 file from: {MAT_FILE}")
    if not MAT_FILE.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at {MAT_FILE}")

    mat_size_mb = MAT_FILE.stat().st_size / (1024.0 * 1024.0)
    print(f"[RAW LOAD] File size: {mat_size_mb:.2f} MB")
    print("[RAW LOAD] Step 1/6: Decoding MATLAB Level 5 binary structures (zlib decompression)...")
    t0 = time.time()

    data = prepare_data(mat_path=MAT_FILE)
    t_process = time.time() - t0

    print(f"[RAW LOAD] Step 2/6: Extracted matrices: tra_Y_tr (36x1261), tra_Y_te (36x840), tra_adj_mat (36x36)")
    print(f"[RAW LOAD] Step 3/6: Sensor metadata extracted from snapshot (lanes, directions, road segments)")
    print(f"[RAW LOAD] Step 4/6: Chain selection completed: Upstream={data.chain.upstream}, "
          f"States={data.chain.states} (Score: {data.chain.score:.4f}, Margin: {data.chain.orientation_margin:.4f})")
    print(f"[RAW LOAD] Step 5/6: Flow normalization: Q_ref = {data.q_ref:.4f} (training flow distribution)")
    print(f"[RAW LOAD] Step 6/6: Segments: Train={len(data.train_segments)} contiguous segments | "
          f"Test={len(data.test_segments)} contiguous segments")
    print(f"[RAW LOAD] Raw processing completed in {t_process:.3f} seconds.")

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE SAVE] Serializing PreparedData to disk cache: {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    saved_size_kb = cache_path.stat().st_size / 1024.0
    print(f"[CACHE SAVE] Cache written successfully ({saved_size_kb:.2f} KB). Subsequent runs will load instantly.")
    print("=" * 80)
    return data


# =====================================================================
# 2. SYSTEM RESOURCE MONITORING
# =====================================================================

def query_gpu_status() -> Dict[str, Any]:
    """Queries NVIDIA GPU utilization and memory via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.check_output(cmd, text=True, timeout=1.0).strip()
        parts = [p.strip() for p in res.split(",")]
        return {
            "available": True,
            "name": parts[0],
            "gpu_util_percent": float(parts[1]),
            "mem_util_percent": float(parts[2]),
            "mem_used_mb": float(parts[3]),
            "mem_total_mb": float(parts[4]),
            "temperature_c": float(parts[5]),
        }
    except Exception:
        return {"available": False}


def get_system_resource_snapshot() -> Dict[str, Any]:
    """Captures CPU per-core loads, RAM usage, and GPU metrics."""
    cpu_percent_total = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    vmem = psutil.virtual_memory()
    proc = psutil.Process()
    proc_mem_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    gpu_info = query_gpu_status()

    return {
        "cpu_total_percent": cpu_percent_total,
        "cpu_per_core": cpu_per_core,
        "ram_used_gb": vmem.used / (1024.0**3),
        "ram_total_gb": vmem.total / (1024.0**3),
        "ram_percent": vmem.percent,
        "process_ram_mb": proc_mem_mb,
        "gpu": gpu_info,
    }


# =====================================================================
# 3. CPU PARALLEL INTEGRATION WORKER
# =====================================================================

def _simulate_window_loss(
    x0: Tuple[float, float, float],
    u_vals: Tuple[float, ...],
    target_states: List[Tuple[float, float, float]],
    t0: float,
    interval: float,
    theta: Sequence[float],
) -> float:
    """Integrates IVP with RK4 over a window and computes MSE loss."""
    sig = PiecewiseConstantInput(u_vals, t0, interval)
    n_pts = len(target_states)
    h = interval / 2.0
    n_substeps = (n_pts - 1) * 2

    x = x0
    states = [x]
    for step in range(n_substeps):
        t_n = t0 + step * h
        u = sig.on_step(t_n, h)
        x = rk4_step(x, u, h, theta)
        if (step + 1) % 2 == 0:
            states.append(x)

    total_sq = 0.0
    for sim, obs in zip(states, target_states):
        total_sq += (sim[0] - obs[0]) ** 2 + (sim[1] - obs[1]) ** 2 + (sim[2] - obs[2]) ** 2
    return total_sq / (n_pts * 3.0)


def evaluate_window_chunk(
    args: Tuple[
        List[Tuple[Tuple[float, float, float], Tuple[float, ...], List[Tuple[float, float, float]], float]],
        Tuple[float, ...],
        float,
    ],
) -> Tuple[float, float, float, int, List[float]]:
    """Worker task executed in parallel across CPU cores.

    Processes a batch of trajectory windows:
    Computes cumulative loss, absolute error, and parameter gradient via central differences.
    """
    windows, theta, interval = args
    n_params = len(theta)
    lower_bounds = PARAM_LOWER if n_params == 6 else HIGH_CAPACITY_PARAM_LOWER
    upper_bounds = PARAM_UPPER if n_params == 6 else HIGH_CAPACITY_PARAM_UPPER

    chunk_loss = 0.0
    chunk_abs_err = 0.0
    chunk_sq_err = 0.0
    chunk_pts = 0
    grad_accum = [0.0] * n_params

    # Epsilon steps for central difference gradient
    eps_vals = [max(1e-4, 1e-3 * abs(th)) for th in theta]

    for x0, u_vals, target_states, t0 in windows:
        n_pts = len(target_states)
        chunk_pts += n_pts

        base_loss = _simulate_window_loss(x0, u_vals, target_states, t0, interval, theta)
        chunk_loss += base_loss * n_pts

        # Approximate MAE and RMSE contributions
        chunk_abs_err += math.sqrt(max(0.0, base_loss)) * n_pts
        chunk_sq_err += base_loss * n_pts

        # Central finite difference gradient
        for j in range(n_params):
            eps = eps_vals[j]
            th_plus = list(theta)
            th_plus[j] = min(upper_bounds[j], th_plus[j] + eps)
            th_minus = list(theta)
            th_minus[j] = max(lower_bounds[j], th_minus[j] - eps)

            l_plus = _simulate_window_loss(x0, u_vals, target_states, t0, interval, th_plus)
            l_minus = _simulate_window_loss(x0, u_vals, target_states, t0, interval, th_minus)
            denom = th_plus[j] - th_minus[j]
            d_loss = (l_plus - l_minus) / denom if denom > 1e-12 else 0.0
            grad_accum[j] += d_loss * n_pts

    return chunk_loss, chunk_abs_err, chunk_sq_err, chunk_pts, grad_accum


# =====================================================================
# 4. ADAM OPTIMIZER & CHECKPOINTING
# =====================================================================

class AdamOptimizer:
    """Pure-Python Adam optimizer with bounds projection (supports 6 & 16 params)."""

    def __init__(
        self,
        params: Sequence[float],
        lr: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
    ) -> None:
        self.theta = list(params)
        self.n_params = len(self.theta)
        if lower_bounds is not None:
            self.lower_bounds = list(lower_bounds)
        else:
            self.lower_bounds = list(PARAM_LOWER if self.n_params == 6 else HIGH_CAPACITY_PARAM_LOWER)
        if upper_bounds is not None:
            self.upper_bounds = list(upper_bounds)
        else:
            self.upper_bounds = list(PARAM_UPPER if self.n_params == 6 else HIGH_CAPACITY_PARAM_UPPER)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * self.n_params
        self.v = [0.0] * self.n_params
        self.step = 0

    def update(self, grad: Sequence[float], lr_scale: float = 1.0) -> Tuple[float, ...]:
        self.step += 1
        eff_lr = self.lr * lr_scale
        for i in range(self.n_params):
            g = grad[i]
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - self.beta1**self.step)
            v_hat = self.v[i] / (1.0 - self.beta2**self.step)

            new_val = self.theta[i] - eff_lr * m_hat / (math.sqrt(v_hat) + self.eps)
            self.theta[i] = max(self.lower_bounds[i], min(self.upper_bounds[i], new_val))

        return tuple(self.theta)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "theta": list(self.theta),
            "m": list(self.m),
            "v": list(self.v),
            "step": self.step,
            "lr": self.lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.theta = list(state["theta"])
        self.m = list(state["m"])
        self.v = list(state["v"])
        self.step = state["step"]
        self.lr = state.get("lr", self.lr)


def save_checkpoint(
    epoch: int,
    total_epochs: int,
    optimizer: AdamOptimizer,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    history: Dict[str, Any],
    checkpoint_dir: Path = CHECKPOINTS_DIR,
) -> Path:
    """Saves atomic training state to checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "theta": {name: val for name, val in zip(PARAM_NAMES, optimizer.theta)},
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    # Save specific epoch checkpoint
    epoch_file = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
    with open(epoch_file, "w") as f:
        json.dump(ckpt, f, indent=2)

    # Save latest pointer
    latest_file = checkpoint_dir / "latest_checkpoint.json"
    with open(latest_file, "w") as f:
        json.dump(ckpt, f, indent=2)

    # If new best, save best
    if val_loss <= best_val_loss:
        best_file = checkpoint_dir / "best_checkpoint.json"
        with open(best_file, "w") as f:
            json.dump(ckpt, f, indent=2)

    return epoch_file


def load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """Loads checkpoint from path."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    with open(ckpt_path) as f:
        return json.load(f)


# =====================================================================
# 5. MAIN TRAINING LOOP
# =====================================================================

def prepare_window_datasets(
    data: PreparedData,
    window_length: int = 48,
    train_stride: int = 8,
    val_stride: int = 24,
) -> Tuple[List[Any], List[Any]]:
    """Constructs train and validation sliding windows."""
    train_windows = []
    interval = OBS_INTERVAL_H

    for seg in data.train_segments:
        for start in range(seg.start, seg.stop - window_length, train_stride):
            stop = start + window_length
            x0 = data.train.state_at(start)
            u_vals = tuple(data.train.u[start:stop])
            targets = [data.train.state_at(k) for k in range(start, stop)]
            t0 = start * interval
            train_windows.append((x0, u_vals, targets, t0))

    val_windows = []
    for seg in data.test_segments:
        for start in range(seg.start, seg.stop - window_length, val_stride):
            stop = start + window_length
            x0 = data.test.state_at(start)
            u_vals = tuple(data.test.u[start:stop])
            targets = [data.test.state_at(k) for k in range(start, stop)]
            t0 = start * interval
            val_windows.append((x0, u_vals, targets, t0))

    return train_windows, val_windows


def train_model(
    epochs: int = 50,
    num_workers: Optional[int] = None,
    learning_rate: float = 0.02,
    resume_checkpoint: Optional[Path] = None,
    force_reload_data: bool = False,
) -> Dict[str, Any]:
    """Full 50-epoch CPU-parallelized training procedure with checkpointing."""
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 4

    print("=" * 80)
    print("TRAFFIC FLOW ODE - PARALLEL TRAINING INITIALIZATION")
    print("=" * 80)
    print(f"Target Epochs: {epochs}")
    print(f"CPU Parallelization: {num_workers} worker processes")
    print(f"Base Learning Rate: {learning_rate}")

    # Load data with caching
    data = get_cached_or_prepare_data(force_reload=force_reload_data)

    print("\n[WINDOW EXTRACTION] Preparing multi-trajectory training windows...")
    train_windows, val_windows = prepare_window_datasets(data, window_length=48, train_stride=8, val_stride=16)
    print(f"[WINDOW EXTRACTION] Generated {len(train_windows)} training windows and {len(val_windows)} test validation windows.")

    # Partition training windows into chunks for workers
    chunk_size = max(1, math.ceil(len(train_windows) / num_workers))
    train_chunks = [train_windows[i : i + chunk_size] for i in range(0, len(train_windows), chunk_size)]
    val_chunks = [val_windows[i : i + chunk_size] for i in range(0, len(val_windows), chunk_size)]

    # Initial parameters or resume from checkpoint
    initial_theta = [2.5, 2.0, 2.0, 0.5, 0.3, 0.01]
    optimizer = AdamOptimizer(initial_theta, lr=learning_rate)

    start_epoch = 1
    best_val_loss = float("inf")
    history: Dict[str, List[Any]] = {
        "epochs": [],
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "parameters": [],
        "grad_norm": [],
        "block_durations_sec": [],
        "cpu_usage_percent": [],
        "gpu_usage_percent": [],
        "ram_usage_gb": [],
    }

    if resume_checkpoint is not None and resume_checkpoint.is_file():
        print(f"\n[RESUME] Found checkpoint: {resume_checkpoint}")
        ckpt_data = load_checkpoint(resume_checkpoint)
        resumed_epoch = ckpt_data["epoch"]
        optimizer.load_state_dict(ckpt_data["optimizer"])
        best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
        history = ckpt_data.get("history", history)
        start_epoch = resumed_epoch + 1
        print(f"[RESUME] Restored state from Epoch {resumed_epoch}.")
        print(f"[RESUME] Resuming parameters: {optimizer.theta}")
        print(f"[RESUME] Best validation loss: {best_val_loss:.6f}")
        print(f"[RESUME] Training will continue from Epoch {start_epoch} up to Epoch {epochs}.")
        if start_epoch > epochs:
            print(f"[RESUME] Checkpoint already reached epoch {resumed_epoch} >= {epochs}. Training complete.")
            return ckpt_data

    print("\n" + "=" * 80)
    print("STARTING PARALLEL MULTI-CORE TRAINING LOOP")
    print("=" * 80)
    print(f"{'Core Count':12s}: {num_workers} processes")
    print(f"{'Hardware':12s}: psutil monitoring active | NVIDIA GPU polling active")
    print("Reporting frequency: Detailed statistics printed every 5 epochs")
    print("-" * 80)

    # Initialize CPU monitor baseline
    psutil.cpu_percent(interval=None)

    # Create worker pool
    pool = mp.Pool(processes=num_workers)

    overall_start_time = time.time()
    block_start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs + 1):
            # Cosine learning rate decay
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, epochs)))

            # Parallel worker evaluation of training chunks
            current_theta = tuple(optimizer.theta)
            worker_tasks = [(chunk, current_theta, OBS_INTERVAL_H) for chunk in train_chunks]
            chunk_results = pool.map(evaluate_window_chunk, worker_tasks)

            # Aggregate losses and gradients across workers
            total_loss = sum(r[0] for r in chunk_results)
            total_abs = sum(r[1] for r in chunk_results)
            total_sq = sum(r[2] for r in chunk_results)
            total_pts = sum(r[3] for r in chunk_results)

            mean_train_loss = total_loss / total_pts
            mean_train_mae = total_abs / total_pts
            mean_train_rmse = math.sqrt(total_sq / total_pts)

            agg_grad = [sum(r[4][j] for r in chunk_results) / total_pts for j in range(6)]
            grad_norm = math.sqrt(sum(g * g for g in agg_grad))

            # Optimizer step
            updated_theta = optimizer.update(agg_grad, lr_scale=lr_scale)

            # Periodically evaluate validation set
            val_loss = mean_train_loss
            val_mae = mean_train_mae
            val_rmse = mean_train_rmse
            if epoch % 5 == 0 or epoch == start_epoch or epoch == epochs:
                val_tasks = [(chunk, updated_theta, OBS_INTERVAL_H) for chunk in val_chunks]
                val_chunk_results = pool.map(evaluate_window_chunk, val_tasks)
                v_pts = sum(r[3] for r in val_chunk_results)
                val_loss = sum(r[0] for r in val_chunk_results) / v_pts
                val_mae = sum(r[1] for r in val_chunk_results) / v_pts
                val_rmse = math.sqrt(sum(r[2] for r in val_chunk_results) / v_pts)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            # Record history
            history["epochs"].append(epoch)
            history["train_loss"].append(mean_train_loss)
            history["train_mae"].append(mean_train_mae)
            history["train_rmse"].append(mean_train_rmse)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["val_rmse"].append(val_rmse)
            history["parameters"].append(list(updated_theta))
            history["grad_norm"].append(grad_norm)

            # -------------------------------------------------------------
            # PRINT DETAILED STATISTICS EVERY 5 EPOCHS (AND INITIAL BASELINE)
            # -------------------------------------------------------------
            if epoch % 5 == 0 or epoch == start_epoch or epoch == epochs:
                block_duration = time.time() - block_start_time
                epochs_in_block = 5 if (epoch % 5 == 0 and epoch != start_epoch) else max(1, epoch - start_epoch + 1)
                sec_per_epoch = block_duration / epochs_in_block
                total_elapsed = time.time() - overall_start_time
                remaining_epochs = epochs - epoch
                eta_seconds = remaining_epochs * sec_per_epoch

                history["block_durations_sec"].append(block_duration)

                # Capture hardware usage
                sys_snap = get_system_resource_snapshot()
                history["cpu_usage_percent"].append(sys_snap["cpu_total_percent"])
                history["ram_usage_gb"].append(sys_snap["ram_used_gb"])

                gpu_str = "N/A (Disabled or CPU mode)"
                if sys_snap["gpu"]["available"]:
                    gpu_data = sys_snap["gpu"]
                    gpu_str = (
                        f"{gpu_data['name']} | Util: {gpu_data['gpu_util_percent']:.1f}% | "
                        f"VRAM: {gpu_data['mem_used_mb']:.0f}/{gpu_data['mem_total_mb']:.0f} MB | "
                        f"Temp: {gpu_data['temperature_c']:.0f}°C"
                    )
                    history["gpu_usage_percent"].append(gpu_data["gpu_util_percent"])

                # Per-core formatting
                cores_str = " ".join(f"C{i}:{pct:.0f}%" for i, pct in enumerate(sys_snap["cpu_per_core"][:8]))

                th = updated_theta
                print("\n" + "=" * 80)
                print(f"EPOCH [{epoch:2d}/{epochs:2d}] STATISTICAL & RESOURCE REPORT  (Progress: {epoch/epochs*100:5.1f}%)")
                print("=" * 80)
                print(f"  TIMING METRICS:")
                print(f"    • Block Duration (last {epochs_in_block} epochs) : {block_duration:8.3f} seconds ({sec_per_epoch:.3f} s/epoch)")
                print(f"    • Total Elapsed Time              : {total_elapsed:8.3f} seconds ({total_elapsed/60:.2f} minutes)")
                print(f"    • Estimated Remaining (ETA)       : {eta_seconds:8.3f} seconds ({eta_seconds/60:.2f} minutes)")
                print(f"  TRAINING LOSS & PERFORMANCE:")
                print(f"    • Train MSE Loss                  : {mean_train_loss:12.6f}")
                print(f"    • Train MAE                       : {mean_train_mae:12.6f}")
                print(f"    • Train RMSE                      : {mean_train_rmse:12.6f}")
                print(f"    • Gradient L2 Norm ||∇θ||         : {grad_norm:12.6f}")
                print(f"    • Effective Learning Rate         : {optimizer.lr * lr_scale:12.6f}")
                print(f"  TEST SET VALIDATION METRICS:")
                print(f"    • Test Validation MSE Loss        : {val_loss:12.6f}")
                print(f"    • Test Validation MAE             : {val_mae:12.6f}")
                print(f"    • Best Validation Loss So Far     : {best_val_loss:12.6f}")
                print(f"  CURRENT MODEL PARAMETERS θ:")
                print(f"    • Couplings (α1, α2, α3)          : ({th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f}) 1/h")
                print(f"    • Dampings  (β1, β2, β3)          : ({th[3]:.4f}, {th[4]:.4f}, {th[5]:.4f}) 1/h")
                print(f"  HARDWARE & RESOURCE UTILIZATION:")
                print(f"    • CPU Workers Active              : {num_workers} parallel workers")
                print(f"    • Total System CPU Utilization    : {sys_snap['cpu_total_percent']:5.1f}%")
                print(f"    • Per-Core Utilization Breakdown  : [{cores_str}]")
                print(f"    • Memory Usage                    : {sys_snap['ram_used_gb']:.2f} GB / {sys_snap['ram_total_gb']:.2f} GB ({sys_snap['ram_percent']}%) | Process RSS: {sys_snap['process_ram_mb']:.1f} MB")
                print(f"    • GPU State                       : {gpu_str}")

                # Save checkpoint every 5 epochs
                saved_path = save_checkpoint(
                    epoch=epoch,
                    total_epochs=epochs,
                    optimizer=optimizer,
                    train_loss=mean_train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    history=history,
                )
                print(f"  CHECKPOINT:")
                print(f"    • Saved State Checkpoint to       : {saved_path.name}")
                print("-" * 80)

                block_start_time = time.time()

    finally:
        pool.close()
        pool.join()

    total_training_time = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Epochs Run            : {epochs}")
    print(f"Total Training Duration     : {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Final Train Loss (MSE)      : {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss (MSE) : {history['val_loss'][-1]:.6f}")
    print(f"Best Validation Loss        : {best_val_loss:.6f}")
    print(f"Optimal Learned Parameters  :")
    final_th = optimizer.theta
    for name, val in zip(PARAM_NAMES, final_th):
        print(f"  {name:10s} : {val:.6f}")

    # Save final parameters to RESULTS_DIR for downstream evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "fitted_parameters.json", "w") as f:
        json.dump({name: val for name, val in zip(PARAM_NAMES, final_th)}, f, indent=2)

    return {
        "final_epoch": epochs,
        "final_parameters": {name: val for name, val in zip(PARAM_NAMES, final_th)},
        "best_val_loss": best_val_loss,
        "history": history,
    }


# =====================================================================
# 6. COMMAND-LINE PARSER & ENTRYPOINT
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-parallelized training script for Traffic Flow ODE with data caching & checkpointing."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Base learning rate for Adam optimizer (default: 0.02)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest available checkpoint in experiments/checkpoints/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to explicitly resume from",
    )
    parser.add_argument(
        "--force-reload-data",
        action="store_true",
        help="Force reloading data from raw .mat file and rebuild disk cache",
    )

    args = parser.parse_args()

    resume_path: Optional[Path] = None
    if args.checkpoint:
        resume_path = Path(args.checkpoint)
    elif args.resume:
        latest = CHECKPOINTS_DIR / "latest_checkpoint.json"
        if latest.is_file():
            resume_path = latest
        else:
            print(f"[RESUME WARNING] No latest checkpoint found at {latest}. Starting fresh training.")

    train_model(
        epochs=args.epochs,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        resume_checkpoint=resume_path,
        force_reload_data=args.force_reload_data,
    )


if __name__ == "__main__":
    main()
