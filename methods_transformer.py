"""TransferabilityMethod wrapper for RankingCrossAttentionTransformer.

Plugs into PARC's Experiment loop. On the first call for a (target_dataset, run)
pair, loads ALL controlled-bank model pkls, runs a single transformer forward pass
over all N models simultaneously, and caches the per-model scores. Subsequent calls
for the same (target, run) return the cached scalar.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make project root importable regardless of cwd (PARC sets cwd=parc/)
# ---------------------------------------------------------------------------
_PARC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PARC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods import TransferabilityMethod  # parc/methods.py


class RankingTransformerMethod(TransferabilityMethod):
    """Score (source_model, target_dataset) pairs using RankingCrossAttentionTransformer.

    The transformer ranks ALL N models in one forward pass.  Since PARC calls
    methods one-by-one per (arch, source, target, run), we cache the full
    target×run score batch on the first call and return cached scalars thereafter.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        probe_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._probe_dir = Path(probe_dir)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None
        # (target_dataset, run, architecture, source_dataset) -> float
        self._score_cache: dict[tuple[str, int, str, str], float] = {}

    # ------------------------------------------------------------------
    # Lazy model loader
    # ------------------------------------------------------------------

    def _get_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        _orig_cwd = os.getcwd()
        os.chdir(_PROJECT_ROOT)
        try:
            from model import RankingCrossAttentionTransformer
            model = RankingCrossAttentionTransformer()
            ckpt = torch.load(
                self._checkpoint_path, map_location=self._device, weights_only=False
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.eval().to(self._device)
            self._model = model
            print(f"  [Transformer] Loaded checkpoint: {self._checkpoint_path.name}")
        finally:
            os.chdir(_orig_cwd)

        return self._model

    # ------------------------------------------------------------------
    # Batch populate: one forward pass for all models on this (target, run)
    # ------------------------------------------------------------------

    def _populate_cache(self, target: str, run: int) -> None:
        """Load all pkls for (target, run), run transformer, cache scores."""
        # Import PARC constants (already importable because cwd=parc/)
        from constants import variables as parc_vars

        archs   = parc_vars["Architecture"]    # ['resnet50', 'resnet18', 'googlenet', 'alexnet']
        sources = parc_vars["Source Dataset"]  # 8 datasets

        model_embs: list[np.ndarray] = []
        dataset_emb: np.ndarray | None = None
        keys: list[tuple[str, int, str, str]] = []

        for arch in archs:
            for source in sources:
                pkl_path = self._probe_dir / f"{arch}_{source}_{target}_{run}.pkl"
                if not pkl_path.exists():
                    continue
                with open(pkl_path, "rb") as f:
                    probe = pickle.load(f)

                if "model_param_embedding" not in probe:
                    continue

                model_embs.append(probe["model_param_embedding"])  # [512]
                keys.append((target, run, arch, source))

                # Use the first available dataset embedding (same target → same images)
                if dataset_emb is None and "clip_image_embedding" in probe:
                    dataset_emb = probe["clip_image_embedding"]    # [500, 512]

        if not model_embs:
            print(f"  [Transformer] No model embeddings found for ({target}, run={run})")
            return
        if dataset_emb is None:
            print(f"  [Transformer] No clip_image_embedding found for ({target}, run={run})")
            return

        # model_tokens:   (1, N, 512)
        # dataset_tokens: (1, 500, 512)
        model_tokens = (
            torch.tensor(np.stack(model_embs), dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )
        dataset_tokens = (
            torch.tensor(dataset_emb, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            # RankingCrossAttentionTransformer.forward(dataset_tokens, model_tokens)
            logits = self._get_model()(dataset_tokens, model_tokens)  # (1, N)

        # Negate logits: the old nn.Transformer with causal decoder masking
        # produces scores where lower = better, but PARC expects higher = better
        scores = -logits[0].cpu().numpy()
        for i, key in enumerate(keys):
            self._score_cache[key] = float(scores[i])

        print(
            f"  [Transformer] Scored {len(keys)} models for "
            f"target={target} run={run} on {self._device}"
        )

    # ------------------------------------------------------------------
    # TransferabilityMethod interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        features,
        probs,
        y,
        source_dataset,
        target_dataset,
        architecture,
        cache_path_fn,
        **kwargs,  # absorb augmented keys (probe_indices, clip_image_embedding, etc.)
    ) -> float:
        self.features = features
        self.probs = probs
        self.y = y
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.architecture = architecture
        self.cache_path_fn = cache_path_fn

        # Extract run number from the pkl path: ../{arch}_{source}_{target}_{run}.pkl
        pkl_path = cache_path_fn(architecture, source_dataset, target_dataset)
        self.run = int(Path(pkl_path).stem.rsplit("_", 1)[-1])

        return self.forward()

    def forward(self) -> float:
        """Return transformer score for the current (arch, source, target, run)."""
        key = (self.target_dataset, self.run, self.architecture, self.source_dataset)
        if key not in self._score_cache:
            self._populate_cache(self.target_dataset, self.run)
        return self._score_cache.get(key, 0.0)
