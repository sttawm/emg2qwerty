# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""KenLM Fusion Layer for reranking CTC beam search hypotheses.

Architecture:
  1. Frozen acoustic model -> emissions (T, N, num_classes)
  2. CTCBeamDecoder -> top-N beam hypotheses per sequence
  3. Per beam: (p_total, lm_cumulative_score, lm_next_char_logits)
  4. Shared MLP per beam (DeepSets-style, permutation invariant via mean aggregation)
  5. Reranking head -> score per hypothesis
  6. Loss: cross-entropy, target = beam with lowest edit distance to ground truth
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance
from torchmetrics import MetricCollection

from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import CTCBeamDecoder
from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty.metrics import CharacterErrorRates

log = logging.getLogger(__name__)


def extract_beam_features(decoder: CTCBeamDecoder, n_beams: int) -> np.ndarray:
    """Extract features from the top-n beams after decoding.

    Per beam:
      - p_total (1): combined acoustic + LM log-probability
      - lm_cumulative (1): sum of raw KenLM step scores for the sequence
      - lm_next_logits (num_classes): KenLM score for each next char given the
        beam's final LM state

    Returns:
        (n_beams, 2 + num_classes) float32 array, zero-padded if fewer beams
        than n_beams are available.
    """
    cs = charset()
    num_classes = cs.num_classes
    features = np.zeros((n_beams, 2 + num_classes), dtype=np.float32)

    for i, beam in enumerate(decoder.beam[:n_beams]):
        p_total = float(beam.p_total)
        lm_cumulative = (
            float(sum(beam.lm_scores)) if beam.lm_node is not None else 0.0
        )

        lm_next = np.zeros(num_classes, dtype=np.float32)
        if decoder.lm is not None and beam.lm_node is not None:
            for label in range(num_classes):
                if label == cs.null_class:
                    continue
                _, score = decoder.apply_lm(beam.lm_state, label)
                lm_next[label] = float(score)

        features[i] = np.concatenate([[p_total, lm_cumulative], lm_next])

    return features


class KenLMFusionLayer(nn.Module):
    """Permutation-invariant fusion layer over beam hypotheses.

    Uses a DeepSets-style architecture:
      1. Shared MLP applied independently to each beam's feature vector
      2. Mean aggregation across beams -> context vector
      3. Concat(beam_embed, context) -> reranking score per beam

    Args:
        num_classes: Vocabulary size including blank token.
        hidden_dim: Hidden dimension for the shared MLP.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        feat_dim = num_classes + 2  # p_total + lm_cumulative + lm_next_logits

        self.beam_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rerank_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, beam_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beam_features: (batch, n_beams, feat_dim)
        Returns:
            scores: (batch, n_beams) reranking logits
        """
        x = self.beam_mlp(beam_features)  # (batch, n_beams, hidden_dim)
        context = x.mean(dim=1, keepdim=True).expand_as(x)  # (batch, n_beams, hidden_dim)
        scores = self.rerank_head(torch.cat([x, context], dim=-1))  # (batch, n_beams, 1)
        return scores.squeeze(-1)  # (batch, n_beams)


class KenLMFusionModule(pl.LightningModule):
    """Trains a KenLM fusion layer on top of a frozen acoustic model.

    The fusion layer reranks the top-N CTC beam search hypotheses, learning
    to combine acoustic and KenLM signals in a permutation-invariant way.
    Only the fusion layer's parameters are trained; the acoustic model is frozen.

    Args:
        acoustic_checkpoint: Path to the frozen acoustic model checkpoint.
        lm_path: Path to the KenLM language model binary.
        n_beams: Number of beam hypotheses to use for fusion (reranking set size).
        hidden_dim: Hidden dimension for the KenLMFusionLayer MLP.
        lr: Learning rate for the fusion layer optimizer.
        beam_size: CTCBeamDecoder beam size. Must be >= n_beams.
    """

    def __init__(
        self,
        acoustic_checkpoint: str,
        lm_path: str,
        n_beams: int = 10,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        beam_size: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_beams = n_beams
        self.lr = lr
        self.lm_path = lm_path
        self.beam_size = beam_size

        # Frozen acoustic model — weights never updated
        self.acoustic_model = TDSConvCTCModule.load_from_checkpoint(
            acoustic_checkpoint
        )
        self.acoustic_model.freeze()

        # Only the fusion layer is trained
        self.fusion = KenLMFusionLayer(
            num_classes=charset().num_classes,
            hidden_dim=hidden_dim,
        )

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        # Lazy init: kenlm.Model is not picklable so can't be set in __init__
        # when using multiprocessing DataLoaders.
        self._decoder: CTCBeamDecoder | None = None

    @property
    def decoder(self) -> CTCBeamDecoder:
        if self._decoder is None:
            self._decoder = CTCBeamDecoder(
                beam_size=self.beam_size,
                lm_path=self.lm_path,
            )
        return self._decoder

    def _get_emissions(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run frozen acoustic model, return (emissions_np, emission_lengths)."""
        with torch.no_grad():
            emissions = self.acoustic_model(inputs)  # (T_out, N, num_classes)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).clamp(min=1)
        return emissions.cpu().numpy(), emission_lengths.cpu().numpy()

    def _decode_batch(
        self,
        emissions_np: np.ndarray,
        emission_lengths: np.ndarray,
    ) -> tuple[list[list[list[int]]], list[np.ndarray]]:
        """Run beam search for each batch item and extract beam features.

        Returns:
            all_hypotheses: list of n_beams decoded label sequences per batch item
            all_features: list of (n_beams, feat_dim) float32 arrays per batch item
        """
        N = emissions_np.shape[1]
        all_hypotheses = []
        all_features = []

        for i in range(N):
            T_i = int(emission_lengths[i])
            self.decoder.reset()
            self.decoder.decode(
                emissions=emissions_np[:T_i, i],
                timestamps=np.arange(T_i),
            )
            all_features.append(extract_beam_features(self.decoder, self.n_beams))
            all_hypotheses.append(
                [list(b.decoding) for b in self.decoder.beam[: self.n_beams]]
            )

        return all_hypotheses, all_features

    def _find_best_beam_indices(
        self,
        all_hypotheses: list[list[list[int]]],
        targets: np.ndarray,
        target_lengths: np.ndarray,
    ) -> list[int]:
        """Find the beam index closest to ground truth by edit distance."""
        cs = charset()
        best_indices = []

        for i, hypotheses in enumerate(all_hypotheses):
            T_i = int(target_lengths[i])
            gt_str = cs.labels_to_str(targets[:T_i, i].tolist())

            best_idx, best_dist = 0, float("inf")
            for j, hyp in enumerate(hypotheses):
                dist = levenshtein_distance(gt_str, cs.labels_to_str(hyp) if hyp else "")
                if dist < best_dist:
                    best_dist, best_idx = dist, j

            best_indices.append(best_idx)

        return best_indices

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        # 1. Emissions from frozen acoustic model
        emissions_np, emission_lengths = self._get_emissions(inputs, input_lengths)

        # 2. Beam search + feature extraction (CPU, non-differentiable)
        all_hypotheses, all_features = self._decode_batch(emissions_np, emission_lengths)

        # 3. Supervision: find which beam is closest to ground truth
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        best_indices = self._find_best_beam_indices(
            all_hypotheses, targets_np, target_lengths_np
        )

        # 4. Fusion layer forward pass (GPU, differentiable)
        beam_features = torch.tensor(
            np.stack(all_features), dtype=torch.float32, device=self.device
        )  # (N, n_beams, feat_dim)
        scores = self.fusion(beam_features)  # (N, n_beams)

        # 5. Cross-entropy: learn to rank the best beam highest
        target_tensor = torch.tensor(best_indices, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(scores, target_tensor)

        # 6. CER for the top-reranked hypothesis
        pred_indices = scores.argmax(dim=1).cpu().tolist()
        metrics = self.metrics[f"{phase}_metrics"]
        cs = charset()
        for i in range(N):
            hyp = all_hypotheses[i][pred_indices[i]] if all_hypotheses[i] else []
            T_i = int(target_lengths_np[i])
            metrics.update(
                prediction=LabelData.from_labels(hyp),
                target=LabelData.from_labels(targets_np[:T_i, i].tolist()),
            )

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.fusion.parameters(), lr=self.lr)
