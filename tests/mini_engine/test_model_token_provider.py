from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest
from llm_opt_lab.mini_engine.token_provider import ModelBackedTokenProvider
from llm_opt_lab.mini_engine.types import TokenIds


class RecordingNextTokenModel:
    def __init__(self, next_token_ids: list[int], vocab_size: int = 16) -> None:
        self.next_token_ids = next_token_ids
        self.vocab_size = vocab_size
        self.forward_inputs: list[TokenIds] = []

    def forward(self, tokens: TokenIds) -> list[list[float]]:
        self.forward_inputs.append(tokens.copy())
        next_token_id = self.next_token_ids[len(self.forward_inputs) - 1]
        logits = [[0.0 for _ in range(self.vocab_size)] for _ in tokens]
        logits[-1][next_token_id] = 1.0
        return logits


class ModelBackedTokenProviderTest(unittest.TestCase):
    def test_returns_highest_logit_token_for_each_active_request(self) -> None:
        model = RecordingNextTokenModel(next_token_ids=[7, 4])
        provider = ModelBackedTokenProvider(model)
        active_requests = [
            ActiveRequest.from_request(GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=2)),
            ActiveRequest.from_request(GenerationRequest("req-2", prompt=[3], max_new_tokens=2)),
        ]

        tokens = provider(active_requests)

        self.assertEqual(tokens, {"req-1": 7, "req-2": 4})
        self.assertEqual(model.forward_inputs, [[1, 2], [3]])

    def test_uses_prompt_plus_generated_tokens_as_model_input(self) -> None:
        model = RecordingNextTokenModel(next_token_ids=[9])
        provider = ModelBackedTokenProvider(model)
        active = ActiveRequest.from_request(GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=3))
        active.append_token(7)

        tokens = provider([active])

        self.assertEqual(tokens, {"req-1": 9})
        self.assertEqual(model.forward_inputs, [[1, 2, 7]])


if __name__ == "__main__":
    unittest.main()
