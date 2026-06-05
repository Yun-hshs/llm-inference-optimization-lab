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
class KVCacheAwareTokenProvider:
    """将模型的token输入分成2个阶段,第一阶段是prefill,第二阶段是decode。
    prefill阶段输入完整的上下文,
    decode阶段每次输入上一个生成的token和更新后的KV cache。"""

    def __init__(self, model: DecoderModel) -> None:
        self.model = model

    def __call__(self, active_requests: list[ActiveRequest]) -> dict[str, int]:
        token_by_request_id = {}
        for active in active_requests:
            if active.phase() == "prefill":
                # Handle prefill phase
                model_input = active.output_tokens()
            else:
                # Handle decode phase
                model_input = [active.generated_tokens[-1]]
            logits = self.model.forward(model_input)
            last_token_logits = logits[-1]
            next_token_id = max(range(len(last_token_logits)), key=lambda i: last_token_logits[i])
            token_by_request_id[active.request.request_id] = next_token_id
            
        return token_by_request_id

