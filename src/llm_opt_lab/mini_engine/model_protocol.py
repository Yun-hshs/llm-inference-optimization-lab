from __future__ import annotations

from typing import Protocol, TypeAlias

from llm_opt_lab.mini_engine.types import TokenIds

Logits: TypeAlias = list[list[float]]


class DecoderModel(Protocol):
    """Minimal model interface used by the educational decoding engine."""

    def forward(self, tokens: TokenIds) -> Logits:
        """Return logits shaped as [sequence_length, vocabulary_size]."""
