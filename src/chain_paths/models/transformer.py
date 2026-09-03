"""Transformer-based predictor wrapper with causal encoder-decoder blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .torch_base import TorchBackedPredictor


@dataclass
class TransformerConfig:
    """Configuration for the causal Transformer predictor."""

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    ff_multiplier: int = 4
    max_seq_len: int = 256
    local_window: int = 5
    global_window: int = -1


class TransformerPredictor(TorchBackedPredictor):
    """Sequence model that honors temporal causality via masked decoding."""

    def __init__(
        self,
        technique_ids: Sequence[str],
        *,
        transition_log_probs: Optional[np.ndarray] = None,
        config: TransformerConfig | None = None,
        tactic_mapping: Optional[Mapping[str, Sequence[str]]] = None,
        tactic_ids: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(technique_ids, transition_log_probs=transition_log_probs)
        self.config = config or TransformerConfig()
        self.model: Any = None
        self._device = None

        tactic_values = tactic_ids or self._extract_tactics_from_mapping(tactic_mapping)
        # Reserve index 0 for unknown/padding tactics to keep masks simple
        self.tactic_ids: Tuple[str, ...] = ("__unknown__",) + tuple(
            sorted({t.strip().lower() for t in tactic_values if str(t).strip()})
        )
        self._tactic_index: Dict[str, int] = {t: idx for idx, t in enumerate(self.tactic_ids)}
        self._technique_tactics: Mapping[str, Sequence[str]] = tactic_mapping or {}

    # ------------------------------------------------------------------
    def _extract_tactics_from_mapping(
        self, mapping: Optional[Mapping[str, Sequence[str]]]
    ) -> Sequence[str]:
        if not mapping:
            return []
        values = []
        for tactics in mapping.values():
            values.extend(tactics)
        return values

    # ------------------------------------------------------------------
    def _tactic_index_for(self, technique: str) -> int:
        tactics = self._technique_tactics.get(technique)
        if not tactics:
            return 0
        normalized = str(tactics[0]).strip().lower()
        return self._tactic_index.get(normalized, 0)

    # ------------------------------------------------------------------
    def _build_model(self):  # pragma: no cover - optional dependency
        """Construct the PyTorch encoder-decoder with multi-scale masking."""

        torch = self._torch

        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
                super().__init__()
                self.dropout = torch.nn.Dropout(dropout)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(torch.log(torch.tensor(10000.0)) / d_model)
                )
                pe = torch.zeros(max_len, d_model, dtype=torch.float)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.pe[:, : x.size(1)]
                return self.dropout(x)

        def _causal_mask(sz: int, device) -> torch.Tensor:
            mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
            return mask

        def _local_causal_mask(sz: int, window: int, device) -> torch.Tensor:
            mask = torch.full((sz, sz), float("-inf"), device=device)
            for i in range(sz):
                start = 0 if window < 0 else max(0, i - window)
                mask[i, start : i + 1] = 0.0
            return mask

        class MultiScaleDecoderLayer(torch.nn.Module):
            def __init__(self, d_model: int, heads: int, dropout: float, ff_multiplier: int, window: int):
                super().__init__()
                local_heads = max(1, heads // 2)
                global_heads = heads - local_heads
                self.self_attn_local = torch.nn.MultiheadAttention(
                    d_model, local_heads, dropout=dropout, batch_first=True
                )
                self.self_attn_global = torch.nn.MultiheadAttention(
                    d_model, max(1, global_heads), dropout=dropout, batch_first=True
                )
                self.cross_attn = torch.nn.MultiheadAttention(
                    d_model, heads, dropout=dropout, batch_first=True
                )
                self.ff = torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_model * ff_multiplier),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(d_model * ff_multiplier, d_model),
                )
                self.norm1 = torch.nn.LayerNorm(d_model)
                self.norm2 = torch.nn.LayerNorm(d_model)
                self.norm3 = torch.nn.LayerNorm(d_model)
                self.dropout = torch.nn.Dropout(dropout)
                self.window = window

            def forward(
                self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                *,
                tgt_mask: torch.Tensor,
                global_mask: torch.Tensor,
                tgt_padding_mask: Optional[torch.Tensor],
                memory_key_padding_mask: Optional[torch.Tensor],
            ) -> torch.Tensor:
                local_mask = tgt_mask if self.window <= 0 else _local_causal_mask(tgt_mask.size(0), self.window, tgt_mask.device)
                local_out, _ = self.self_attn_local(
                    tgt, tgt, tgt, attn_mask=local_mask, key_padding_mask=tgt_padding_mask
                )
                global_out, _ = self.self_attn_global(
                    tgt, tgt, tgt, attn_mask=global_mask, key_padding_mask=tgt_padding_mask
                )
                tgt2 = self.norm1(tgt + self.dropout(0.5 * (local_out + global_out)))
                cross_out, _ = self.cross_attn(
                    tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
                )
                tgt3 = self.norm2(tgt2 + self.dropout(cross_out))
                ff_out = self.ff(tgt3)
                return self.norm3(tgt3 + self.dropout(ff_out))

        class CausalTransformer(torch.nn.Module):
            def __init__(self, vocab: int, tactic_vocab: int, cfg: TransformerConfig):
                super().__init__()
                self.tech_embed = torch.nn.Embedding(vocab, cfg.d_model)
                self.tactic_embed = torch.nn.Embedding(tactic_vocab, cfg.d_model)
                self.positional = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    cfg.d_model,
                    cfg.num_heads,
                    cfg.d_model * cfg.ff_multiplier,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
                self.decoders = torch.nn.ModuleList(
                    [
                        MultiScaleDecoderLayer(
                            cfg.d_model,
                            cfg.num_heads,
                            cfg.dropout,
                            cfg.ff_multiplier,
                            cfg.local_window,
                        )
                        for _ in range(cfg.num_layers)
                    ]
                )
                self.output = torch.nn.Linear(cfg.d_model, vocab)
                self.cfg = cfg

            def forward(
                self,
                tgt_indices: torch.Tensor,
                tactic_indices: Optional[torch.Tensor] = None,
                *,
                tgt_padding_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                # If tactic_indices not provided, create dummy tactic indices
                # Map all techniques to tactic index 0 (unknown/padding)
                if tactic_indices is None:
                    tactic_indices = torch.zeros_like(tgt_indices)
                
                tgt = self.positional(self.tech_embed(tgt_indices))
                memory = self.positional(self.tactic_embed(tactic_indices))
                memory_key_padding = tactic_indices.eq(0) if tactic_indices.numel() else None
                memory_encoded = self.encoder(memory, src_key_padding_mask=memory_key_padding)

                seq_len = tgt.size(1)
                causal_mask = _causal_mask(seq_len, tgt.device)
                global_mask = causal_mask if self.cfg.global_window < 0 else _local_causal_mask(
                    seq_len, self.cfg.global_window, tgt.device
                )
                out = tgt
                for layer in self.decoders:
                    out = layer(
                        out,
                        memory_encoded,
                        tgt_mask=causal_mask,
                        global_mask=global_mask,
                        tgt_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=memory_key_padding,
                    )
                logits = self.output(out)
                return logits

        model = CausalTransformer(len(self.technique_ids), len(self.tactic_ids), self.config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(self._device)

    # ------------------------------------------------------------------
    def augment_features(
        self, history: Sequence[str], tactic_history: Sequence[str]
    ) -> Dict[str, Any]:
        """Expose model-ready tensors to the feature extractor pipeline."""

        tech_indices = [self._index.get(t, 0) for t in history][-self.config.max_seq_len :]
        tactic_indices = [self._tactic_index.get(t, 0) for t in tactic_history][-self.config.max_seq_len :]
        return {
            "transformer": {
                "technique_indices": tech_indices,
                "tactic_indices": tactic_indices,
                "config": self.config,
            }
        }

    # ------------------------------------------------------------------
    def _encode_history(self, history: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        tech_indices = [self._index.get(t, 0) for t in history][-self.config.max_seq_len :]
        tactic_indices = [self._tactic_index_for(t) for t in history][-self.config.max_seq_len :]
        return np.asarray(tech_indices, dtype=np.int64), np.asarray(tactic_indices, dtype=np.int64)

    # ------------------------------------------------------------------
    def base_log_probabilities(self, history, candidates, group_id=None):  # type: ignore[override]
        if not self._torch_available:
            return super().base_log_probabilities(history, candidates, group_id)

        if self.model is None:
            self.model = self._build_model()

        torch = self._torch
        self.model.eval()

        tech_indices, tactic_indices = self._encode_history(history)
        if tech_indices.size == 0:
            tech_indices = np.array([0], dtype=np.int64)
            tactic_indices = np.array([0], dtype=np.int64)

        with torch.no_grad():
            tech_tensor = torch.as_tensor(tech_indices, device=self._device).unsqueeze(0)
            tactic_tensor = torch.as_tensor(tactic_indices, device=self._device).unsqueeze(0)
            tgt_padding_mask = tech_tensor.eq(0)
            logits = self.model(tech_tensor, tactic_tensor, tgt_padding_mask=tgt_padding_mask)
            step_logits = logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)

        indices = self.ensure_candidate_order(candidates)
        selected = log_probs[0, indices].detach().cpu().numpy().astype(np.float64)
        diagnostics = [
            {
                "p_count": float(np.exp(lp)),
                "source": "transformer",
                "uses_causal_mask": True,
            }
            for lp in selected
        ]
        return selected, diagnostics

    # ------------------------------------------------------------------
    def get_model_name(self) -> str:
        """Return model identifier."""
        return 'transformer'


__all__ = ["TransformerPredictor", "TransformerConfig"]
