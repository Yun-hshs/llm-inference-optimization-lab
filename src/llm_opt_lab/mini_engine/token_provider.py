from __future__ import annotations

from llm_opt_lab.mini_engine.model_protocol import DecoderModel
from llm_opt_lab.mini_engine.scheduler import ActiveRequest


class ModelBackedTokenProvider:
    """Select one next token for each active request using a DecoderModel."""

    def __init__(self, model: DecoderModel) -> None:
        self.model = model

    def __call__(self, active_requests: list[ActiveRequest]) -> dict[str, int]:
        # Handwrite the model-backed greedy token selection here.
        next_token_dict = {}
        for active in active_requests:
            output_tokens = active.output_tokens()
            logits = self.model.forward(output_tokens)
            last_token_logits = logits[-1]
            next_token_id = max(range(len(last_token_logits)), key=lambda i: last_token_logits[i])
            #在schedule中以及新加的，这里不要加
            #active.append_token(next_token_id)
            next_token_dict[active.request.request_id] = next_token_id
        return next_token_dict
