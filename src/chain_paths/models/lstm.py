"""Unidirectional LSTM-based sequence predictor (causal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .torch_base import TorchBackedPredictor


@dataclass
class LSTMConfig:
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 256


class LSTMPredictor(TorchBackedPredictor):
    """Causal LSTM predictor (left-to-right)."""

    def __init__(
        self,
        technique_ids: Sequence[str],
        *,
        transition_log_probs: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        tech_to_idx: Optional[Dict[str, int]] = None,
        config: LSTMConfig | None = None,
    ) -> None:
        super().__init__(technique_ids, transition_log_probs=transition_log_probs)
        self.embeddings = embeddings
        self.tech_to_idx = tech_to_idx or {}
        self.config = config or LSTMConfig()

    def get_model_name(self) -> str:
        return 'lstm'

    def _build_model(self):  # pragma: no cover - optional dependency
        if not self._torch_available:
            return None

        torch = self._torch
        vocab_size = len(self.technique_ids)

        if self.embeddings is not None:
            embed_dim = int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 128
            weight = torch.randn(vocab_size, embed_dim, dtype=torch.float32) * 0.02
            for idx, tech in enumerate(self.technique_ids):
                src_idx = self.tech_to_idx.get(tech)
                if src_idx is None:
                    continue
                if 0 <= src_idx < self.embeddings.shape[0]:
                    weight[idx] = torch.tensor(self.embeddings[src_idx], dtype=torch.float32)
            embedding = torch.nn.Embedding.from_pretrained(weight, freeze=False)
        else:
            embed_dim = 128
            embedding = torch.nn.Embedding(vocab_size, embed_dim)

        lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )

        class _LSTM(torch.nn.Module):
            def __init__(self, emb, lstm_module, hidden_size, dropout, vocab):
                super().__init__()
                self.embedding = emb
                self.lstm = lstm_module
                self.dropout = torch.nn.Dropout(dropout)
                self.output = torch.nn.Linear(hidden_size, vocab)

            def forward(self, x):
                emb = self.embedding(x)
                output, _ = self.lstm(emb)
                output = self.dropout(output)
                return self.output(output)

        return _LSTM(embedding, lstm, self.config.hidden_size, self.config.dropout, len(self.technique_ids))

    def _encode_history(self, history: Sequence[str]) -> np.ndarray:
        max_len = getattr(self.config, "max_seq_len", 256)
        indices = [self._index.get(t, 0) for t in history][-max_len:]
        if not indices:
            indices = [0]
        return np.asarray(indices, dtype=np.int64)

    def base_log_probabilities(self, history, candidates, group_id=None):  # type: ignore[override]
        if not self._torch_available:
            return super().base_log_probabilities(history, candidates, group_id)

        if self.model is None:
            self.model = self._build_model()

        if self.model is None:
            return super().base_log_probabilities(history, candidates, group_id)

        torch = self._torch
        self.model.eval()

        hist_indices = self._encode_history(history)
        device = next(self.model.parameters()).device
        hist_tensor = torch.as_tensor(hist_indices, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(hist_tensor)
            step_logits = logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)

        indices = self.ensure_candidate_order(candidates)
        if indices.size == 0:
            return np.array([], dtype=np.float64), []

        selected = log_probs[0, indices].detach().cpu().numpy().astype(np.float64)
        diagnostics = [
            {
                "p_count": float(np.exp(lp)),
                "source": "lstm",
            }
            for lp in selected
        ]
        return selected, diagnostics


__all__ = ["LSTMPredictor", "LSTMConfig"]
