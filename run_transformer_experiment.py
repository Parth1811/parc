"""Run PARC benchmark with all methods + RankingCrossAttentionTransformer.

Evaluates all methods with both Pearson correlation (PARC default) and
Kendall's tau (τ_a, τ_b, τ_w) + Spearman ρ.

Usage
-----
# Run all methods + transformer:
    python parc/run_transformer_experiment.py \
        --checkpoint artifacts/models/transformer/checkpoint_RankingCrossAttentionTransformer_epoch_50_20251117_083539.pt \
        --device cuda --name all_methods

# Evaluate only (results CSV already exists):
    python parc/run_transformer_experiment.py \
        --checkpoint dummy \
        --eval-only results/all_methods.csv

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

from evaluate import Experiment
from methods import PARC, LEEP, NegativeCrossEntropy, HScore, kNN, RSA, DDS
from methods_transformer import RankingTransformerMethod
from evaluate_ranking import evaluate_ranking, RankingMetrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PARC benchmark with all methods + RankingTransformer."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to RankingCrossAttentionTransformer checkpoint (.pt)",
    )
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p.add_argument(
        "--name",
        default="all_methods",
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
# Build methods dict
# ---------------------------------------------------------------------------

def build_methods(checkpoint_path: Path, probe_dir: Path, device: str) -> dict:
    return {
        # --- Transformer (ours) ---
        "RankingTransformer": RankingTransformerMethod(
            checkpoint_path=checkpoint_path,
            probe_dir=probe_dir,
            device=device,
        ),
        # --- PARC baselines ---
        "PARC f=32": PARC(n_dims=32),
        "PARC f=64": PARC(n_dims=64),
        "LEEP": LEEP(),
        "LEEP f=32": LEEP(n_dims=32, src="features"),
        "NCE": NegativeCrossEntropy(),
        "HScore": HScore(),
        "HScore f=32": HScore(n_dims=32),
        "1-NN CV": kNN(k=1),
        "3-NN CV": kNN(k=3),
        "RSA": RSA(),
        "DDS": DDS(),
    }


# ---------------------------------------------------------------------------
# Kendall's tau evaluation for all methods
# ---------------------------------------------------------------------------

def evaluate_all_methods_kendall(
    results_csv: str | Path,
    oracle_csv: str | Path,
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

    # Detect method columns (everything after the 4 key columns)
    key_cols = {"Run", "Architecture", "Source Dataset", "Target Dataset", "Oracle"}
    method_cols = [c for c in merged.columns if c not in key_cols]

    print(f"\nResults: {len(results)} rows, Oracle: {len(oracle)} rows, Merged: {len(merged)} rows")
    print(f"Methods found: {method_cols}")

    # --- Per-method overall Kendall tau ---
    print(f"\n{'Method':<22} {'τ_a':>8} {'τ_b':>8} {'τ_w':>8} {'ρ':>8} {'Pearson':>8}")
    print("=" * 78)

    for method_col in method_cols:
        if merged[method_col].isna().all():
            continue

        all_metrics: list[RankingMetrics] = []
        pearson_vals: list[float] = []

        for (target, run), group in merged.groupby(["Target Dataset", "Run"]):
            group = group[group["Source Dataset"] != group["Target Dataset"]]
            if len(group) < 2:
                continue

            pred_scores = group[method_col].to_numpy(dtype=float)
            true_scores = group["Oracle"].to_numpy(dtype=float)

            # Skip if all predictions are the same (can't rank)
            if np.std(pred_scores) < 1e-10:
                continue

            pred_ranks = np.argsort(np.argsort(-pred_scores))
            true_ranks = np.argsort(np.argsort(-true_scores))

            m = evaluate_ranking(true_ranks.tolist(), pred_ranks.tolist())
            all_metrics.append(m)

            # Pearson correlation (PARC's default metric)
            from scipy import stats
            valid = true_scores > 0
            if valid.sum() >= 2:
                r, _ = stats.pearsonr(pred_scores[valid], true_scores[valid])
                pearson_vals.append(r * 100)

        if all_metrics:
            print(
                f"{method_col:<22} "
                f"{np.mean([m.kendall_tau_a for m in all_metrics]):>8.4f} "
                f"{np.mean([m.kendall_tau_b for m in all_metrics]):>8.4f} "
                f"{np.mean([m.kendall_tau_w for m in all_metrics]):>8.4f} "
                f"{np.mean([m.spearman_rho  for m in all_metrics]):>8.4f} "
                f"{np.mean(pearson_vals) if pearson_vals else float('nan'):>8.2f}"
            )

    # --- Per-target breakdown for each method ---
    print(f"\n{'='*90}")
    print("PER-TARGET BREAKDOWN")
    print(f"{'='*90}")

    for method_col in method_cols:
        if merged[method_col].isna().all():
            continue

        print(f"\n--- {method_col} ---")
        print(f"{'Target Dataset':<20} {'τ_a':>8} {'τ_b':>8} {'τ_w':>8} {'ρ':>8} {'n_runs':>7}")
        print("-" * 68)

        by_target: dict[str, list[RankingMetrics]] = defaultdict(list)
        for (target, run), group in merged.groupby(["Target Dataset", "Run"]):
            group = group[group["Source Dataset"] != group["Target Dataset"]]
            if len(group) < 2:
                continue

            pred_scores = group[method_col].to_numpy(dtype=float)
            true_scores = group["Oracle"].to_numpy(dtype=float)

            if np.std(pred_scores) < 1e-10:
                continue

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if args.eval_only:
        results_csv = Path(args.eval_only)
        if not results_csv.is_absolute():
            results_csv = PARC_DIR / results_csv
        print(f"Evaluating existing results: {results_csv}")
        evaluate_all_methods_kendall(results_csv, args.oracle)
        return

    # --- Run PARC experiment with all methods ---
    probe_dir = PARC_DIR / "cache" / "probes" / "fixed_budget_500"
    methods = build_methods(checkpoint_path, probe_dir, args.device)

    print(f"Running PARC experiment: {args.name}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {args.device}")
    print(f"  Probe dir  : {probe_dir}")
    print(f"  Methods    : {list(methods.keys())}")
    print(f"  Append     : {args.append}")

    experiment = Experiment(methods, name=args.name, append=args.append)
    experiment.run()

    # --- Evaluate all methods ---
    results_csv = Path(experiment.out_file)
    print(f"\nExperiment complete. Results: {results_csv}")
    evaluate_all_methods_kendall(results_csv, args.oracle)


if __name__ == "__main__":
    main()
