from __future__ import annotations

from llm_opt_lab.mini_engine.model_protocol import DecoderModel
from llm_opt_lab.mini_engine.types import TokenIds


class GreedyEngine:
    """A tiny greedy decoding engine for learning LLM inference flow."""

    def __init__(self, model: DecoderModel) -> None:
        self.model = model

    def generate(
        self,
        prompt: TokenIds,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> TokenIds:
        #raise NotImplementedError("Handwrite the greedy decoding loop here.")
        #开始实现功能
        #步骤一：初始化输出为输入的prompt
        output_token_ids = prompt.copy()  # 初始化输出为输入的prompt
        #步骤二：进入循环，直到生成max_new_tokens个新token或者遇到eos_token_id
        for _ in range(max_new_tokens):
            #步骤三：调用模型的forward方法，获取下一个token的概率分布的分数
            logits = self.model.forward(output_token_ids)
            #步骤四：从概率分布中选择概率最高的token作为下一个token
            last_token_logits = logits[-1]
            # 选择概率最高的token
            next_token_id = max(range(len(last_token_logits)), key=lambda i: last_token_logits[i])
            #步骤五：将下一个token添加到输出中
            output_token_ids.append(next_token_id)
            #步骤六：如果下一个token是eos_token_id，则停止生成
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
        return output_token_ids