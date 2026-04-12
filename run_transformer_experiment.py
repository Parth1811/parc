"""Run PARC benchmark with RankingCrossAttentionTransformer and evaluate with Kendall's tau.

Usage
-----
# Run experiment + evaluate:
    python parc/run_transformer_experiment.py \
        --checkpoint artifacts/models/transformer/RankingCrossAttentionTransformer_best_<ts>.pt \
        --device cuda \
        --name transformer_experiment \
        --append

# Evaluate only (results CSV already exists):
    python parc/run_transformer_experiment.py \
        --checkpoint <any path> \
        --eval-only results/transformer_experiment.csv

Run from the project root so all imports resolve correctly.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — must happen before any project/parc imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARC_DIR = PROJECT_ROOT / "parc"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PARC_DIR))

# PARC uses relative paths (./cache/, ./results/, ./oracles/) so cwd must be parc/
os.chdir(PARC_DIR)

from evaluate import Experiment          # parc/evaluate.py
from methods_transformer import RankingTransformerMethod
from evaluate_ranking import evaluate_ranking, RankingMetrics   # project root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RankingCrossAttentionTransformer on the PARC benchmark."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to RankingCrossAttentionTransformer checkpoint (.pt)",
    )
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p.add_argument(
        "--name",
        default="transformer_experiment",
        help="Experiment name (results saved to parc/results/<name>.csv)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Resume from existing results CSV instead of overwriting",
    )
    p.add_argument(
        "--eval-only",
        metavar="RESULTS_CSV",
        default=None,
        help="Skip experiment, evaluate an existing results CSV with Kendall's tau",
    )
    p.add_argument(
        "--oracle",
        default="./oracles/controlled.csv",
        help="Path to oracle CSV (default: parc/oracles/controlled.csv)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Kendall's tau evaluation
# ---------------------------------------------------------------------------

def evaluate_with_kendall(
    results_csv: str | Path,
    oracle_csv: str | Path,
    method_col: str = "RankingTransformer",
) -> None:
    results = pd.read_csv(results_csv)
    oracle  = pd.read_csv(oracle_csv)

    merged = results.merge(
        oracle,
        on=["Architecture", "Source Dataset", "Target Dataset"],
        how="inner",
    )

    if merged.empty:
        print("[ERROR] Merge produced empty dataframe — check column names.")
        return

    print(f"\nResults: {len(results)} rows, Oracle: {len(oracle)} rows, Merged: {len(merged)} rows")
    print(f"\n{'Target Dataset':<20} {'τ_a':>8} {'τ_b':>8} {'τ_w':>8} {'ρ':>8} {'n_runs':>7}")
    print("-" * 68)

    by_target: dict[str, list[RankingMetrics]] = defaultdict(list)

    for (target, run), group in merged.groupby(["Target Dataset", "Run"]):
        # Skip source==target pairs (PARC skips these too)
        group = group[group["Source Dataset"] != group["Target Dataset"]]
        if len(group) < 2:
            continue

        pred_scores = group[method_col].to_numpy(dtype=float)
        true_scores = group["Oracle"].to_numpy(dtype=float)

        # Convert scores to rank positions (0 = best)
        pred_ranks = np.argsort(np.argsort(-pred_scores))
        true_ranks = np.argsort(np.argsort(-true_scores))

        m = evaluate_ranking(true_ranks.tolist(), pred_ranks.tolist())
        by_target[target].append(m)

    overall: list[RankingMetrics] = []
    for target, ms in sorted(by_target.items()):
        print(
            f"{target:<20} "
            f"{np.mean([m.kendall_tau_a for m in ms]):>8.4f} "
            f"{np.mean([m.kendall_tau_b for m in ms]):>8.4f} "
            f"{np.mean([m.kendall_tau_w for m in ms]):>8.4f} "
            f"{np.mean([m.spearman_rho  for m in ms]):>8.4f} "
            f"{len(ms):>7}"
        )
        overall.extend(ms)

    if overall:
        print("-" * 68)
        print(
            f"{'OVERALL':<20} "
            f"{np.mean([m.kendall_tau_a for m in overall]):>8.4f} "
            f"{np.mean([m.kendall_tau_b for m in overall]):>8.4f} "
            f"{np.mean([m.kendall_tau_w for m in overall]):>8.4f} "
            f"{np.mean([m.spearman_rho  for m in overall]):>8.4f} "
            f"{len(overall):>7}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if args.eval_only:
        # Skip experiment, just evaluate existing CSV
        results_csv = Path(args.eval_only)
        if not results_csv.is_absolute():
            results_csv = PARC_DIR / results_csv
        print(f"Evaluating existing results: {results_csv}")
        evaluate_with_kendall(results_csv, args.oracle)
        return

    # --- Run PARC experiment ---
    probe_dir = PARC_DIR / "cache" / "probes" / "fixed_budget_500"
    methods = {
        "RankingTransformer": RankingTransformerMethod(
            checkpoint_path=checkpoint_path,
            probe_dir=probe_dir,
            device=args.device,
        ),
    }

    print(f"Running PARC experiment: {args.name}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {args.device}")
    print(f"  Probe dir  : {probe_dir}")
    print(f"  Append     : {args.append}")

    experiment = Experiment(methods, name=args.name, append=args.append)
    experiment.run()

    # --- Kendall's tau evaluation ---
    results_csv = Path(experiment.out_file)
    print(f"\nExperiment complete. Results: {results_csv}")
    evaluate_with_kendall(results_csv, args.oracle)


if __name__ == "__main__":
    main()
