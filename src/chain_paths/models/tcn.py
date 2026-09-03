"""Temporal convolutional network predictor wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .torch_base import TorchBackedPredictor


@dataclass
class TCNConfig:
    channels: int = 256
    depth: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256


class TCNPredictor(TorchBackedPredictor):
    def __init__(
        self,
        technique_ids: Sequence[str],
        *,
        transition_log_probs: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        tech_to_idx: Optional[Dict[str, int]] = None,
        config: TCNConfig | None = None,
    ) -> None:
        super().__init__(technique_ids, transition_log_probs=transition_log_probs)
        self.embeddings = embeddings
        self.tech_to_idx = tech_to_idx or {}
        self.config = config or TCNConfig()

    def get_model_name(self) -> str:
        """Return model identifier."""
        return 'tcn'

    # ------------------------------------------------------------------
    def _build_model(self):  # pragma: no cover - optional dependency
        """Construct the TCN classifier module if PyTorch is available."""

        if not self._torch_available:
            return None

        torch = self._torch

        vocab_size = len(self.technique_ids)

        # Build (or randomly initialize) an embedding table aligned to the model vocab
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

        class _TemporalConvBlock(torch.nn.Module):
            """Single dilated causal convolution block with residual connection."""
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
                super().__init__()
                # For causal convolution: pad on left only to look at past
                # Right padding = 0, left padding = (kernel_size - 1) * dilation
                self.padding = (kernel_size - 1) * dilation
                self.conv = torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=0,  # We'll do manual padding
                )
                self.dropout = torch.nn.Dropout(dropout)
                self.relu = torch.nn.ReLU()
                # Residual projection if dimensions change
                self.residual_proj = (
                    torch.nn.Conv1d(in_channels, out_channels, 1)
                    if in_channels != out_channels
                    else None
                )

            def forward(self, x):
                # x shape: (batch, channels, seq_len)
                residual = x if self.residual_proj is None else self.residual_proj(x)
                
                # Apply causal padding (left padding only)
                x_padded = torch.nn.functional.pad(x, (self.padding, 0))
                
                # Apply convolution without internal padding
                out = self.conv(x_padded)
                
                out = self.relu(out)
                out = self.dropout(out)
                
                # Add residual connection
                return out + residual

        class _TCN(torch.nn.Module):
            def __init__(self, emb, embed_dim, channels, depth, dropout, vocab):
                super().__init__()
                self.embedding = emb
                self.dropout = torch.nn.Dropout(dropout)
                
                # Project embedding dimension to TCN channel dimension
                self.proj = torch.nn.Linear(embed_dim, channels)
                
                # Build stacked dilated convolution blocks
                self.tcn_blocks = torch.nn.ModuleList()
                for layer in range(depth):
                    # Exponential dilation: 1, 2, 4, 8, ...
                    dilation = 2 ** layer
                    self.tcn_blocks.append(
                        _TemporalConvBlock(
                            channels,
                            channels,
                            kernel_size=3,
                            dilation=dilation,
                            dropout=dropout,
                        )
                    )
                
                # Output projection to vocabulary
                self.output = torch.nn.Linear(channels, vocab)

            def forward(self, x):
                # x shape: (batch, seq_len)
                emb = self.embedding(x)  # (batch, seq_len, embed_dim)
                emb = self.dropout(emb)
                
                # Project to TCN channel dimension
                proj = self.proj(emb)  # (batch, seq_len, channels)
                
                # Transpose for conv1d (expects batch, channels, seq_len)
                h = proj.transpose(1, 2)  # (batch, channels, seq_len)
                
                # Apply TCN blocks
                for block in self.tcn_blocks:
                    h = block(h)
                
                # Transpose back to (batch, seq_len, channels)
                h = h.transpose(1, 2)
                
                # Project to vocabulary
                logits = self.output(h)  # (batch, seq_len, vocab)
                return logits

        return _TCN(
            embedding,
            embed_dim,
            self.config.channels,
            self.config.depth,
            self.config.dropout,
            vocab_size
        )

    # ------------------------------------------------------------------
    def _encode_history(self, history: Sequence[str]) -> np.ndarray:
        max_len = getattr(self.config, "max_seq_len", 256)
        indices = [self._index.get(t, 0) for t in history][-max_len:]
        if not indices:
            indices = [0]
        return np.asarray(indices, dtype=np.int64)

    # ------------------------------------------------------------------
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
                "source": "tcn",
            }
            for lp in selected
        ]
        return selected, diagnostics


__all__ = ["TCNPredictor", "TCNConfig"]
