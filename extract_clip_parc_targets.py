"""Extract CLIP image embeddings for the 7 PARC target datasets.

Produces shards in artifacts/extracted/datasets/<slug>/<split>/<split>_clip_NNNNN.npz
matching the format produced by evaluate_clip_imagenet.py.

Shard format:
    features:     (batches_per_shard, num_classes, 512) float32
    class_ids:    (batches_per_shard, num_classes)     int64
    class_names:  (batches_per_shard, num_classes)     object (str)
    actual_batches, target_batches: shape (1,) int32

Each batch in a shard contains exactly one sample per class (class-balanced).

Usage:
    python parc/extract_clip_parc_targets.py
    python parc/extract_clip_parc_targets.py --datasets cub200 nabird
    python parc/extract_clip_parc_targets.py --skip-existing
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARC_DIR = PROJECT_ROOT / "parc"

# Must insert project root BEFORE chdir so ClipImageEncoder and config are importable
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PARC_DIR))

from config import ConfigParser, ClipEvaluationConfig, DatasetLoaderDefaultsConfig
from model import ClipImageEncoder
from dataloader.imagenet_dataset import ClassBalancedBatchSampler

# PARC modules (need cwd = parc/ at import time for their data paths)
_orig_cwd = os.getcwd()
os.chdir(PARC_DIR)
from datasets import construct_dataset, get_dataset_path  # parc/datasets.py
os.chdir(_orig_cwd)


# ---------------------------------------------------------------------------
# PARC target datasets and their output slugs
# ---------------------------------------------------------------------------

# Map PARC internal name -> output directory slug (matches evaluate_clip_imagenet's _make_slug)
PARC_TARGETS: dict[str, str] = {
    "cifar10":       "cifar_10",
    "oxford_pets":   "oxford_pets",
    "cub200":        "cub200",
    "caltech101":    "caltech_101",
    "stanford_dogs": "stanford_dogs",
    "nabird":        "nabird",
    "voc2007":       "voc2007",
}

# Split → PARC train flag mapping
# PARC datasets only expose train/test. We use:
#   train      = PARC train split
#   test       = PARC test split
#   validation = (skipped — no val split available in PARC loaders)
SPLITS: dict[str, bool] = {
    "train": True,
    "test":  False,
}


# ---------------------------------------------------------------------------
# Shard writing (matches evaluate_clip_imagenet._emit_shard)
# ---------------------------------------------------------------------------

def _save_shard(
    shard_path: Path,
    features: np.ndarray,
    class_ids: np.ndarray,
    class_names: np.ndarray,
    *,
    actual_batches: int,
    target_batches: int,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_path,
        features=features,
        class_ids=class_ids,
        class_names=class_names,
        actual_batches=np.array([actual_batches], dtype=np.int32),
        target_batches=np.array([target_batches], dtype=np.int32),
    )


def _emit_shard(
    buffer: list[dict[str, np.ndarray]],
    *,
    shard_size: int,
    shard_dir: Path,
    split_name: str,
    shard_index: int,
    pad_to_full: bool,
) -> int:
    if not buffer:
        return shard_index

    emit_count = min(shard_size, len(buffer))
    entries = buffer[:emit_count]
    del buffer[:emit_count]

    features    = np.stack([e["features"]    for e in entries], axis=0)
    class_ids   = np.stack([e["class_ids"]   for e in entries], axis=0)
    class_names = np.stack([e["class_names"] for e in entries], axis=0)

    actual_batches = features.shape[0]
    target_batches = shard_size

    if pad_to_full and actual_batches < shard_size:
        num_classes = features.shape[1]
        embed_dim   = features.shape[2]
        pad_batches = shard_size - actual_batches
        features = np.concatenate(
            [features, np.zeros((pad_batches, num_classes, embed_dim), dtype=features.dtype)], axis=0
        )
        class_ids = np.concatenate(
            [class_ids, np.full((pad_batches, num_classes), -1, dtype=class_ids.dtype)], axis=0
        )
        class_names = np.concatenate(
            [class_names, np.full((pad_batches, num_classes), "", dtype=object)], axis=0
        )

    shard_path = shard_dir / f"{split_name}_clip_{shard_index:05d}.npz"
    _save_shard(shard_path, features, class_ids, class_names,
                actual_batches=actual_batches, target_batches=target_batches)
    print(f"    saved {shard_path.name}  shape={tuple(features.shape)}  actual={actual_batches}")
    return shard_index + 1


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def _collect_labels(torch_dataset) -> list[int]:
    """Return list of int labels in dataset order (works for PARC datasets)."""
    # PARC datasets expose different label attributes — try common ones
    for attr in ("labels", "targets", "_labels"):
        if hasattr(torch_dataset, attr):
            values = getattr(torch_dataset, attr)
            try:
                return [int(v) for v in values]
            except (TypeError, ValueError):
                pass

    # Fallback: iterate the dataset, pulling only label (slow for large datasets but one-time)
    print(f"    [warn] no label attribute found, iterating dataset to collect labels...")
    labels = []
    for i in range(len(torch_dataset)):
        item = torch_dataset[i]
        # PARC returns (img, target) or (img, target, idx)
        target = item[1]
        # VOC2007 returns multi-label vector — use first nonzero class as primary label
        if hasattr(target, "__len__") and not isinstance(target, (int, str)):
            arr = np.asarray(target).flatten()
            nz = np.flatnonzero(arr)
            labels.append(int(nz[0]) if len(nz) > 0 else 0)
        else:
            labels.append(int(target))
    return labels


def process_dataset(
    parc_name: str,
    slug: str,
    split_name: str,
    is_train: bool,
    *,
    clip_encoder: ClipImageEncoder,
    output_root: Path,
    eval_config: ClipEvaluationConfig,
    defaults: DatasetLoaderDefaultsConfig,
    skip_existing: bool,
) -> None:
    split_dir = output_root / slug / split_name
    if skip_existing and split_dir.exists() and any(split_dir.glob("*.npz")):
        print(f"  [SKIP] {slug}/{split_name} (already has shards)")
        return

    # Construct dataset with CLIP transform
    # PARC's construct_dataset uses its own test_transforms; we override with CLIP's transform
    transform = clip_encoder.build_transform(train=is_train)

    _cwd = os.getcwd()
    os.chdir(PARC_DIR)  # PARC datasets use relative './data/' paths
    try:
        # Import PARC's dataset_objs registry directly so we can pass our own transform
        from datasets import dataset_objs
        dataset_cls = dataset_objs[parc_name]
        torch_dataset = dataset_cls(
            get_dataset_path(parc_name), is_train, transform=transform
        )
    finally:
        os.chdir(_cwd)

    labels = _collect_labels(torch_dataset)
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)

    print(f"  {slug}/{split_name}: {len(torch_dataset)} samples, {num_classes} classes")

    # Class-balanced sampler (one sample per class per batch)
    sampler = ClassBalancedBatchSampler(
        labels, drop_last=defaults.drop_last, shuffle=defaults.shuffle, seed=defaults.seed
    )

    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_sampler=sampler,
        num_workers=defaults.num_workers,
        pin_memory=defaults.pin_memory,
        persistent_workers=defaults.persistent_workers if defaults.num_workers > 0 else False,
        prefetch_factor=defaults.prefetch_factor if defaults.num_workers > 0 else None,
    )

    # Class id -> human readable name (PARC datasets don't always have class_names, fallback to str(idx))
    class_names_map = {int(c): str(c) for c in unique_labels}
    if hasattr(torch_dataset, "class_names"):
        cnames = torch_dataset.class_names
        class_names_map = {int(i): str(n) for i, n in enumerate(cnames)}

    batches_target = math.ceil(len(torch_dataset) / max(num_classes, 1))
    if eval_config.limit_batches_per_split is not None:
        batches_target = min(batches_target, int(eval_config.limit_batches_per_split))

    shard_size = eval_config.batches_per_shard if eval_config.batches_per_shard > 0 else batches_target
    pad_to_full = bool(eval_config.pad_to_full_shard)

    split_dir.mkdir(parents=True, exist_ok=True)

    batch_buffer: list[dict[str, np.ndarray]] = []
    shard_index = 0
    batches_processed = 0

    for batch in loader:
        if batches_processed >= batches_target:
            break

        # PARC returns (img, target) or (img, target, idx)
        if len(batch) >= 2:
            images = batch[0]
            targets = batch[1]
        else:
            raise ValueError(f"Unexpected batch structure: {type(batch)}")

        # Handle multi-label (VOC2007): take first nonzero class
        if targets.dim() > 1:
            targets_np = targets.numpy()
            primary = []
            for row in targets_np:
                nz = np.flatnonzero(row)
                primary.append(int(nz[0]) if len(nz) > 0 else 0)
            labels_np = np.asarray(primary, dtype=np.int64)
        else:
            labels_np = targets.cpu().numpy().astype(np.int64)

        features = clip_encoder.encode(images)
        features_np = features.cpu().numpy()

        class_names_np = np.array(
            [class_names_map.get(int(l), str(l)) for l in labels_np], dtype=object
        )

        batch_buffer.append({
            "features": features_np,
            "class_ids": labels_np,
            "class_names": class_names_np,
        })
        batches_processed += 1

        if shard_size > 0 and len(batch_buffer) >= shard_size:
            shard_index = _emit_shard(
                batch_buffer, shard_size=shard_size, shard_dir=split_dir,
                split_name=split_name, shard_index=shard_index, pad_to_full=False,
            )

    if batch_buffer:
        final_shard_size = shard_size if shard_size > 0 and pad_to_full else len(batch_buffer)
        final_pad = shard_size > 0 and pad_to_full
        shard_index = _emit_shard(
            batch_buffer, shard_size=final_shard_size, shard_dir=split_dir,
            split_name=split_name, shard_index=shard_index, pad_to_full=final_pad,
        )

    print(f"  Done: {slug}/{split_name}  batches={batches_processed}  shards={shard_index}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Subset of PARC datasets to extract. Default: all. Choices: {list(PARC_TARGETS)}",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS.keys()),
        help=f"Splits to extract. Default: {list(SPLITS.keys())}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (slug, split) combinations that already have shard files.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    defaults    = ConfigParser.get(DatasetLoaderDefaultsConfig)

    clip_encoder = ClipImageEncoder(
        model_name=eval_config.model_name,
        device=eval_config.device,
        precision=eval_config.precision,
        normalize_features=eval_config.normalize_features,
    )

    output_root = Path(eval_config.output_directory).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    targets = args.datasets or list(PARC_TARGETS.keys())
    for parc_name in targets:
        if parc_name not in PARC_TARGETS:
            print(f"[WARN] unknown dataset '{parc_name}', skipping")
            continue
        slug = PARC_TARGETS[parc_name]
        print(f"\n=== {parc_name} (slug: {slug}) ===")
        for split_name in args.splits:
            if split_name not in SPLITS:
                print(f"  [WARN] unknown split '{split_name}', skipping")
                continue
            is_train = SPLITS[split_name]
            try:
                process_dataset(
                    parc_name, slug, split_name, is_train,
                    clip_encoder=clip_encoder,
                    output_root=output_root,
                    eval_config=eval_config,
                    defaults=defaults,
                    skip_existing=args.skip_existing,
                )
            except Exception as e:
                import traceback
                print(f"  [ERROR] {slug}/{split_name}: {e}")
                print(traceback.format_exc())


if __name__ == "__main__":
    main()
