from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.engine import GreedyEngine
from llm_opt_lab.mini_engine.types import TokenIds


class ConstantNextTokenModel:
    def __init__(self, next_token_id: int, vocab_size: int = 16) -> None:
        self.next_token_id = next_token_id
        self.vocab_size = vocab_size

    def forward(self, tokens: TokenIds) -> list[list[float]]:
        logits = [[0.0 for _ in range(self.vocab_size)] for _ in tokens]
        logits[-1][self.next_token_id] = 1.0
        return logits


class SequentialNextTokenModel:
    def __init__(self, next_token_ids: list[int], vocab_size: int = 16) -> None:
        self.next_token_ids = next_token_ids
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, tokens: TokenIds) -> list[list[float]]:
        next_token_id = self.next_token_ids[self.forward_calls]
        self.forward_calls += 1

        logits = [[0.0 for _ in range(self.vocab_size)] for _ in tokens]
        logits[-1][next_token_id] = 1.0
        return logits


class GreedyEngineTest(unittest.TestCase):
    def test_generate_appends_highest_logit_token_once(self) -> None:
        engine = GreedyEngine(ConstantNextTokenModel(next_token_id=7))

        generated = engine.generate([1, 2, 3], max_new_tokens=1)

        self.assertEqual(generated, [1, 2, 3, 7])

    def test_generate_appends_multiple_tokens_one_step_at_a_time(self) -> None:
        model = SequentialNextTokenModel([7, 8, 9])
        engine = GreedyEngine(model)

        generated = engine.generate([1, 2, 3], max_new_tokens=3)

        self.assertEqual(generated, [1, 2, 3, 7, 8, 9])
        self.assertEqual(model.forward_calls, 3)

    def test_generate_stops_when_eos_token_is_generated(self) -> None:
        model = SequentialNextTokenModel([7, 2, 9])
        engine = GreedyEngine(model)

        generated = engine.generate([1, 2, 3], max_new_tokens=3, eos_token_id=2)

        self.assertEqual(generated, [1, 2, 3, 7, 2])
        self.assertEqual(model.forward_calls, 2)


if __name__ == "__main__":
    unittest.main()
