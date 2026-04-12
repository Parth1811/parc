"""Extract parameter vectors for all 32 PARC controlled-bank models.

For models found on HuggingFace, downloads and uses those weights.
For models not found (18/28 non-imagenet combinations), falls back to
torchvision imagenet pretrained weights.

Saves one .npz per model to artifacts/extracted/models-faiss/{name}.npz
with key 'parameters' (float32, shape [8192]).

Usage:
    python parc/extract_parc_models.py
    python parc/extract_parc_models.py --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    AlexNet_Weights,
    GoogLeNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "extracted" / "models-faiss"
OUTPUT_SIZE = 8192

# torchvision imagenet weights for each arch
_IMAGENET_WEIGHTS = {
    "alexnet":   AlexNet_Weights.IMAGENET1K_V1,
    "resnet18":  ResNet18_Weights.IMAGENET1K_V1,
    "resnet50":  ResNet50_Weights.IMAGENET1K_V2,
    "googlenet": GoogLeNet_Weights.IMAGENET1K_V1,
}

# num_classes for HuggingFace models (non-imagenet)
_NUM_CLASSES = {
    "cifar10":       10,
    "caltech101":    101,
    "oxford_pets":   37,
    "cub200":        200,
    "stanford_dogs": 120,
    "nabird":        555,
    "voc2007":       21,
}


def load_model_torchvision_imagenet(arch: str) -> nn.Module:
    weights = _IMAGENET_WEIGHTS[arch]
    model = getattr(tv_models, arch)(weights=weights)
    return model.eval()


def load_model_from_hf(arch: str, hf_model_id: str, num_classes: int) -> nn.Module:
    """Download model weights from HuggingFace and load into torchvision arch."""
    from huggingface_hub import hf_hub_download
    import json

    print(f"  Downloading {hf_model_id} from HuggingFace ...")

    kwargs = {"aux_logits": False, "init_weights": False} if arch == "googlenet" else {}
    model = getattr(tv_models, arch)(weights=None, num_classes=num_classes, **kwargs)

    # Try common weight file names
    weight_files = ["pytorch_model.bin", "model.pth", "model.pt", "weights.pth",
                    "best_model.pth", "model_weights.pth", f"{arch}.pth",
                    f"{arch}_{hf_model_id.split('/')[-1]}.pth"]

    state_dict = None
    for fname in weight_files:
        try:
            local_path = hf_hub_download(repo_id=hf_model_id, filename=fname)
            ckpt = torch.load(local_path, map_location="cpu")
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt
            print(f"  Loaded weights from {fname}")
            break
        except Exception:
            continue

    if state_dict is None:
        # Try listing repo files
        try:
            from huggingface_hub import list_repo_files
            files = list(list_repo_files(hf_model_id))
            pt_files = [f for f in files if f.endswith((".bin", ".pth", ".pt", ".safetensors"))]
            print(f"  Available weight files: {pt_files}")
            for fname in pt_files:
                try:
                    local_path = hf_hub_download(repo_id=hf_model_id, filename=fname)
                    if fname.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(local_path)
                    else:
                        ckpt = torch.load(local_path, map_location="cpu")
                        if isinstance(ckpt, dict):
                            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                        else:
                            state_dict = ckpt
                    print(f"  Loaded weights from {fname}")
                    break
                except Exception as e:
                    print(f"  Failed {fname}: {e}")
        except Exception as e:
            raise RuntimeError(f"Could not download weights from {hf_model_id}: {e}")

    if state_dict is None:
        raise RuntimeError(f"No loadable weight file found in {hf_model_id}")

    # Strip DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def extract_param_vector(model: nn.Module, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """Flatten all weight parameters, truncate or pad to output_size."""
    parts = []
    for tensor in model.state_dict().values():
        parts.append(tensor.detach().cpu().numpy().flatten())
    flat = np.concatenate(parts).astype(np.float32)
    if flat.size >= output_size:
        return flat[:output_size]
    return np.pad(flat, (0, output_size - flat.size))


def save_npz(name: str, vector: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.npz"
    np.savez_compressed(path, parameters=vector)
    return path


def load_csv() -> list[dict]:
    csv_path = Path(__file__).parent / "models.csv"
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = load_csv()
    ok = skipped = failed = 0

    for row in rows:
        arch = row["arch"]
        source = row["source_dataset"]
        hf_id = row["huggingface_model_id"]
        num_classes = int(row["num_classes"]) if row["num_classes"] else 1000
        name = f"{arch}_{source}"
        out_path = OUTPUT_DIR / f"{name}.npz"

        if args.skip_existing and out_path.exists():
            print(f"[SKIP] {name}")
            skipped += 1
            continue

        print(f"[EXTRACT] {name}  (hf: {hf_id})")

        try:
            if hf_id == "torchvision/imagenet":
                model = load_model_torchvision_imagenet(arch)
            else:
                try:
                    model = load_model_from_hf(arch, hf_id, num_classes)
                except Exception as e:
                    print(f"  [WARN] HuggingFace load failed ({e}), falling back to imagenet weights")
                    model = load_model_torchvision_imagenet(arch)

            vector = extract_param_vector(model)
            path = save_npz(name, vector)
            print(f"  [OK] saved {path.name}  shape={vector.shape}")
            ok += 1

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed += 1

    print(f"\nDone: {ok} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
