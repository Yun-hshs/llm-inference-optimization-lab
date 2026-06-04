from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler
from llm_opt_lab.mini_engine.serving_loop import ServingLoop
from llm_opt_lab.mini_engine.token_provider import ModelBackedTokenProvider
from llm_opt_lab.mini_engine.types import TokenIds


class SequentialLogitModel:
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


class ModelServingLoopTest(unittest.TestCase):
    def test_serving_loop_uses_model_provider_one_token_per_step(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))
        model = SequentialLogitModel(next_token_ids=[7, 8])
        token_provider = ModelBackedTokenProvider(model)
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        finished = loop.run_until_idle()

        self.assertEqual([active.output_tokens() for active in finished], [[1, 7, 8]])
        self.assertEqual(model.forward_inputs, [[1], [1, 7]])
        self.assertTrue(scheduler.is_idle())


if __name__ == "__main__":
    unittest.main()
