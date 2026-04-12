"""Augment PARC probe cache files with CLIP image embeddings and model parameter embeddings.

Usage
-----
# Verify that sampling reproduces the exact same labels as a cached pkl (dry-run):
    python parc/augment_probes.py --verify-only

# Augment a single probe file:
    python parc/augment_probes.py --arch alexnet --source imagenet --target cifar10 --run 0

# Augment all cached probes (skips files that already have the new keys):
    python parc/augment_probes.py --all --skip-existing

Run from the project root so that all imports resolve correctly.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so that `parc.*` and top-level
# modules (model, extractors, config) all import cleanly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "parc"))

# Import project modules first while cwd=PROJECT_ROOT so ConfigParser finds config.ini
from model.clip_encoder import ClipImageEncoder
from extractors.torchvision_extractor import TorchvisionModelExtractor

# PARC's dataset helpers use relative paths (./data/, ./cache/) so they must
# be called with parc/ as the working directory.
PARC_DIR = PROJECT_ROOT / "parc"
os.chdir(PARC_DIR)

import constants as parc_constants                        # parc/constants.py
from datasets import DatasetCache, FixedBudgetSampler     # parc/datasets.py
from utils import seed_all                                 # parc/utils.py

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROBE_DIR   = PROJECT_ROOT / "parc" / "cache" / "probes" / "fixed_budget_500"
MODELS_DIR  = PROJECT_ROOT / "parc" / "models"

# ---------------------------------------------------------------------------
# Constants mirroring parc/constants.py
# ---------------------------------------------------------------------------
CONTROLLED_ARCHS    = ["resnet50", "resnet18", "googlenet", "alexnet"]
CONTROLLED_SOURCES  = [
    "nabird", "oxford_pets", "cub200", "caltech101",
    "stanford_dogs", "voc2007", "cifar10", "imagenet",
]
TARGET_DATASETS = [
    "cifar10", "oxford_pets", "cub200", "caltech101",
    "stanford_dogs", "nabird", "voc2007",
]
PROBE_SIZE = 500
NUM_RUNS   = 5

# ---------------------------------------------------------------------------
# Phase 1: reproduce probe sampling
# ---------------------------------------------------------------------------

def get_probe_indices(target_dataset: str, run: int) -> list[int]:
    """Reproduce PARC's exact FixedBudgetSampler selection.

    seed_all() must be called immediately before FixedBudgetSampler.__init__
    so that the internal random.shuffle() calls consume the correct state.
    """
    seed_all(2020 + run * 3037)
    loader = FixedBudgetSampler(
        target_dataset,
        batch_size=128,
        probe_size=PROBE_SIZE,
        train=True,
        pin_memory=False,
    )
    return list(loader.sampler)


# ---------------------------------------------------------------------------
# Phase 2: CLIP image embeddings
# ---------------------------------------------------------------------------

_clip_encoder: ClipImageEncoder | None = None
_autoencoder: torch.nn.Module | None = None

def _get_clip_encoder() -> ClipImageEncoder:
    global _clip_encoder
    if _clip_encoder is None:
        _clip_encoder = ClipImageEncoder()
    return _clip_encoder


def _get_autoencoder() -> torch.nn.Module | None:
    """Load the autoencoder once from the config-specified checkpoint."""
    global _autoencoder
    if _autoencoder is not None:
        return _autoencoder

    # Must run from PROJECT_ROOT so ConfigParser finds config.ini
    _orig_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        from model.model_encoder import ModelAutoEncoder

        import configparser as _cp
        _parser = _cp.ConfigParser()
        _parser.read(PROJECT_ROOT / "config.ini")
        raw_path = _parser.get("autoencoder_evaluation", "weights_path", fallback=None)

        checkpoint_path = None
        if raw_path:
            candidate = PROJECT_ROOT / raw_path
            if candidate.exists():
                checkpoint_path = candidate

        if checkpoint_path is None:
            artifact_dir = PROJECT_ROOT / "artifacts" / "models" / "model_autoencoder"
            candidates = sorted(artifact_dir.glob("ModelAutoEncoder_best_*.pt"))
            if candidates:
                checkpoint_path = candidates[-1]

        if checkpoint_path is None:
            return None

        print(f"  [AE] Loading autoencoder: {checkpoint_path.name}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        ae = ModelAutoEncoder()
        ae.load_state_dict(state_dict)
        ae.eval()
        _autoencoder = ae
        return _autoencoder
    except Exception as e:
        print(f"  [AE] Failed to load autoencoder: {e}")
        return None
    finally:
        os.chdir(_orig_cwd)


def get_clip_embeddings(target_dataset: str, indices: list[int]) -> np.ndarray:
    """Return CLIP embeddings for the 500 probe images.

    DatasetCache stores images already transformed with PARC's test_transforms
    (resized + normalized with ImageNet/CIFAR stats).  We pass those tensors
    directly to CLIP — note that PARC's normalization differs from CLIP's, so
    the resulting embeddings capture the dataset's visual content as-seen
    by the PARC pipeline rather than canonical CLIP embeddings.

    Shape: (500, 512)
    """
    cache = DatasetCache(target_dataset, train=True)
    images = torch.stack([cache[i][0] for i in indices])   # [500, 3, 224, 224]
    clip = _get_clip_encoder()
    batch_size = 64
    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(clip.device)
            all_embs.append(clip.encode(batch).cpu())
    return torch.cat(all_embs, dim=0).numpy()   # [500, 512]


# ---------------------------------------------------------------------------
# Phase 3: model parameter embedding  (8192 → via autoencoder → 512)
# ---------------------------------------------------------------------------

def get_model_param_vector(
    arch: str,
    source: str,
    pth_path: Path | None = None,
    output_size: int = 8192,
) -> np.ndarray:
    """Return a fixed-size parameter vector for a torchvision model.

    Instead of FAISS K-means (which can segfault on small layers), we concatenate
    all weight parameters, then truncate or pad to output_size.
    """
    # BaseExtractor.__init__ calls ConfigParser which needs cwd=PROJECT_ROOT
    _orig_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        extractor = TorchvisionModelExtractor(arch, source, pth_path=pth_path)
        params = extractor.load_parameters()   # list of (N, 1) float32 arrays
    finally:
        os.chdir(_orig_cwd)
    flat = np.concatenate([p.flatten() for p in params]).astype(np.float32)
    if flat.size >= output_size:
        return flat[:output_size]
    return np.pad(flat, (0, output_size - flat.size))


def compress_with_autoencoder(raw: np.ndarray) -> np.ndarray:
    """Compress an 8192-dim parameter vector to 512 dims using the trained autoencoder."""
    ae = _get_autoencoder()
    if ae is None:
        raise FileNotFoundError(
            "No autoencoder checkpoint found. Train with: python train.py ModelAutoEncoderTrainer"
        )
    tensor = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)   # [1, 8192]
    with torch.no_grad():
        encoded, _ = ae(tensor)   # encoded: [1, 512]
    return encoded.squeeze(0).cpu().numpy()   # [512]


def get_model_embedding(
    arch: str,
    source: str,
    pth_path: Path | None,
    use_autoencoder: bool = True,
) -> np.ndarray:
    """Return compressed 512-dim model embedding, or raw 8192-dim if no autoencoder."""
    raw = get_model_param_vector(arch, source, pth_path)
    if use_autoencoder:
        return compress_with_autoencoder(raw)
    return raw


# ---------------------------------------------------------------------------
# PARC .pth path resolution
# ---------------------------------------------------------------------------

def resolve_pth_path(arch: str, source: str) -> Path | None:
    """Return path to PARC .pth file, or None if not needed (imagenet) or missing."""
    if source == "imagenet":
        return None   # torchvision pretrained weights are used
    pth = MODELS_DIR / arch / f"{arch}_{source}.pth"
    if pth.exists():
        return pth
    return None   # will raise inside TorchvisionModelExtractor if needed


# ---------------------------------------------------------------------------
# Core augmentation function
# ---------------------------------------------------------------------------

def augment_probe(
    arch: str,
    source: str,
    target: str,
    run: int,
    *,
    verify: bool = True,
    skip_existing: bool = False,
    use_autoencoder: bool = True,
) -> bool:
    """Augment one probe pkl with clip_image_embedding, model_param_embedding, probe_indices.

    Returns True on success, False if skipped.
    """
    pkl_path = PROBE_DIR / f"{arch}_{source}_{target}_{run}.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] pkl not found: {pkl_path.name}")
        return False

    with open(pkl_path, "rb") as f:
        probe = pickle.load(f)

    if skip_existing and "clip_image_embedding" in probe and "model_param_embedding" in probe:
        print(f"  [SKIP] already augmented: {pkl_path.name}")
        return False

    print(f"  Processing {pkl_path.name} ...")

    # ---- Phase 1: reproduce indices (skip if CLIP already present) ----
    clip_emb: np.ndarray | None = None
    indices: list[int] | None = None

    if "clip_image_embedding" in probe:
        clip_emb = probe["clip_image_embedding"]
        print(f"  [SKIP-CLIP] using cached clip_image_embedding")
    else:
        try:
            indices = get_probe_indices(target, run)
        except Exception as e:
            print(f"  [ERROR] sampling failed: {e}")
            return False

        # ---- Verify labels match ----
        if verify:
            try:
                cache = DatasetCache(target, train=True)
                y_check = np.array([cache[i][1] for i in indices])
                if not np.array_equal(y_check, probe["y"]):
                    mismatches = int((y_check != probe["y"]).sum())
                    print(f"  [ERROR] label mismatch ({mismatches}/500). Sampling not reproduced.")
                    return False
                print(f"  [OK] labels verified")
            except Exception as e:
                print(f"  [ERROR] verification failed: {e}")
                return False

        # ---- Phase 2: CLIP embeddings ----
        try:
            clip_emb = get_clip_embeddings(target, indices)   # [500, 512]
            print(f"  [OK] clip_image_embedding {clip_emb.shape}")
        except Exception as e:
            print(f"  [ERROR] CLIP embedding failed: {e}")
            return False

    # ---- Phase 3: model param embedding ----
    pth_path = resolve_pth_path(arch, source)
    model_emb: np.ndarray | None = None
    if source == "imagenet" or pth_path is not None:
        try:
            model_emb = get_model_embedding(arch, source, pth_path, use_autoencoder=use_autoencoder)
            print(f"  [OK] model_param_embedding {model_emb.shape}")
        except FileNotFoundError as e:
            print(f"  [WARN] model embedding skipped: {e}")
        except Exception as e:
            print(f"  [WARN] model embedding failed: {e}")
    else:
        print(f"  [WARN] No .pth for ({arch}, {source}) — model_param_embedding skipped")

    # ---- Augment pkl in-place ----
    if indices is not None:
        probe["probe_indices"] = np.array(indices, dtype=np.int64)
    probe["clip_image_embedding"] = clip_emb
    if model_emb is not None:
        probe["model_param_embedding"] = model_emb

    with open(pkl_path, "wb") as f:
        pickle.dump(probe, f)

    print(f"  [DONE] saved {pkl_path.name}")
    return True


# ---------------------------------------------------------------------------
# Iterator over all controlled probe triples
# ---------------------------------------------------------------------------

def iter_controlled_probes() -> Iterator[tuple[str, str, str, int]]:
    """Yield (arch, source, target, run) for the 4-arch controlled bank."""
    for arch in CONTROLLED_ARCHS:
        for source in CONTROLLED_SOURCES:
            for target in TARGET_DATASETS:
                for run in range(NUM_RUNS):
                    yield arch, source, target, run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment PARC probe pkl files with CLIP and model embeddings."
    )
    p.add_argument("--arch",   default="alexnet",  help="Model architecture")
    p.add_argument("--source", default="imagenet", help="Source dataset the model was trained on")
    p.add_argument("--target", default="cifar10",  help="Target dataset used for probing")
    p.add_argument("--run",    type=int, default=0, help="Run index (0-4)")

    p.add_argument(
        "--all",
        action="store_true",
        help="Augment all controlled-bank probes (4 archs × 8 sources × 7 targets × 5 runs)",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that sampling reproduces correct labels; do not write anything",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip probe files that already contain clip_image_embedding",
    )
    p.add_argument(
        "--no-autoencoder",
        action="store_true",
        help="Store raw 8192-dim param vector instead of autoencoder-compressed 512-dim",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_ae = not args.no_autoencoder

    if args.verify_only:
        # Quick dry-run: just check label reproducibility for first example
        arch, source, target, run = args.arch, args.source, args.target, args.run
        pkl_path = PROBE_DIR / f"{arch}_{source}_{target}_{run}.pkl"
        print(f"Verifying sampling for {pkl_path.name} ...")

        with open(pkl_path, "rb") as f:
            probe = pickle.load(f)

        indices = get_probe_indices(target, run)
        cache   = DatasetCache(target, train=True)
        y_check = np.array([cache[i][1] for i in indices])

        if np.array_equal(y_check, probe["y"]):
            print(f"[OK] Labels match perfectly (500/500)")
            print(f"     y[:10] = {y_check[:10].tolist()}")
        else:
            n = int((y_check != probe["y"]).sum())
            print(f"[FAIL] {n}/500 label mismatches")
            print(f"  Expected y[:10]: {probe['y'][:10].tolist()}")
            print(f"  Got      y[:10]: {y_check[:10].tolist()}")
        return

    if args.all:
        total = ok = skipped = failed = 0
        for arch, source, target, run in iter_controlled_probes():
            total += 1
            result = augment_probe(
                arch, source, target, run,
                verify=True,
                skip_existing=args.skip_existing,
                use_autoencoder=use_ae,
            )
            if result:
                ok += 1
            else:
                skipped += 1
        print(f"\nDone: {ok} augmented, {skipped} skipped, out of {total} total")
    else:
        augment_probe(
            args.arch, args.source, args.target, args.run,
            verify=True,
            skip_existing=args.skip_existing,
            use_autoencoder=use_ae,
        )


if __name__ == "__main__":
    main()
