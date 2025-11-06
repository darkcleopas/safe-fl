#!/usr/bin/env python3
import argparse
import os
import logging
import gc
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# IMPORTANT: set environment knobs BEFORE importing TensorFlow to avoid oneDNN/OpenMP issues
# Defaults are conservative to prevent segfaults when mixing TF with Python threads
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# Limit native thread pools (OpenMP/BLAS) to avoid oversubscription and crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Disable oneDNN optimizations by default as they sometimes segfault under heavy threading
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from fl_simulator import FLSimulator  # or SimpleFLSimulator

logging.getLogger("tensorflow").setLevel(logging.ERROR)

def _configure_tf_threads(tf_threads: int) -> None:
    """Apply TensorFlow threading limits for this process.
    Keep values low when using Python-level threading to avoid instability.
    """
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        # Some TF builds may not support changing threads at runtime; ignore
        pass


def _run_single_config(config_file: str, use_threads: bool, tf_threads: int) -> str:
    # Isolated runner for multiprocessing
    # If using Python threads inside the simulation, restrict TF native threads
    # to prevent oversubscription/segfaults.
    _configure_tf_threads(tf_threads)
    simulator = FLSimulator(config_file, use_threads=use_threads)
    simulator.run_simulation()
    del simulator
    gc.collect()
    tf.keras.backend.clear_session()
    return config_file


def main():
    parser = argparse.ArgumentParser(description='Run federated learning simulation')
    parser.add_argument('--config', help='Path to config file or config directory or config file list', type=str, required=True, nargs='+')
    parser.add_argument('--threads', action='store_true', help='Use multi-threading inside each simulation (Python threads)')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel processes to run configs (per-process isolation)')
    parser.add_argument('--tf-threads', type=int, default=None, help='Override TensorFlow intra/inter op thread count for each run')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress/ETA output')
    args = parser.parse_args()

    config_files = []
    for config in args.config:
        if not os.path.exists(config):
            raise ValueError(f"Config path does not exist: {config}")
        elif os.path.isdir(config):
            config_files.extend([os.path.join(config, f) for f in os.listdir(config) if f.endswith('.yaml')])
        elif os.path.isfile(config):
            if not config.endswith('.yaml'):
                raise ValueError(f"Provided config file is not a YAML file: {config}")
            config_files.append(config)


    # Decide safe TF thread count per run
    # If using Python threads inside FLSimulator, keep TF native threads at 1 for stability.
    # Otherwise, scale by available CPUs divided by jobs (at least 1), which avoids oversubscription.
    if args.threads:
        default_tf_threads = 1
    else:
        cpu_count = os.cpu_count() or 4
        per_proc = max(1, cpu_count // max(1, args.jobs))
        # Cap to a small number to be conservative with TF native pools
        default_tf_threads = min(per_proc, 4)
    tf_threads = args.tf_threads if args.tf_threads is not None else default_tf_threads

    show_progress = not args.no_progress
    total = len(config_files)
    if args.jobs and args.jobs > 1 and total > 1:
        # Run in parallel processes
        # Use 'spawn' context to avoid forking TensorFlow state, which is safer and prevents crashes
        ctx = mp.get_context("spawn")
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as executor:
            futures = [executor.submit(_run_single_config, cf, args.threads, tf_threads) for cf in config_files]
            done = 0
            durations = []
            for fut in as_completed(futures):
                try:
                    done_cfg = fut.result()
                    print(f"Finished simulation for config: {done_cfg}")
                except Exception as e:
                    done_cfg = None
                    logging.exception("A simulation failed")
                done += 1
                if show_progress:
                    elapsed = time.time() - start
                    # Use average duration per finished unit to estimate remaining
                    avg = elapsed / max(1, done)
                    remaining = max(0.0, total - done) * avg
                    eta_hours = int(remaining // 3600)
                    eta_mins = int((remaining % 3600) // 60)
                    eta_secs = int(remaining % 60)
                    pct = (done / total) * 100.0
                    print(f"Progress: {done}/{total} ({pct:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {eta_hours}h {eta_mins}m {eta_secs}s")
    else:
        # Sequential fallback
        start = time.time()
        for i, config_file in enumerate(config_files, start=1):
            cfg_start = time.time()
            _run_single_config(config_file, args.threads, tf_threads)
            print(f"Finished simulation for config: {config_file}")
            if show_progress:
                elapsed = time.time() - start
                avg = elapsed / i
                remaining = max(0.0, total - i) * avg
                eta_hours = int(remaining // 3600)
                eta_mins = int((remaining % 3600) // 60)
                eta_secs = int(remaining % 60)
                pct = (i / total) * 100.0
                print(f"Progress: {i}/{total} ({pct:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {eta_hours}h {eta_mins}m {eta_secs}s")

    print("Simulation completed!")

if __name__ == "__main__":
    main()