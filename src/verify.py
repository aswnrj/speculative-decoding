from typing import Optional
import torch
from transformers import DynamicCache, PreTrainedModel

from src.sampling import SamplingConfig

@torch.inference_mode()
def verify_target(
    model: PreTrainedModel,
    prefix_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    sampling: SamplingConfig,
    past_key_values: Optional[DynamicCache] = None,
) -> tuple[torch.Tensor, DynamicCache]:
    if past_key_values is None:
        past_key_values = DynamicCache()

    assert prefix_ids.shape[1] >= 1

    full_input = torch.cat([prefix_ids, draft_tokens], dim=-1)
    out = model(input_ids=full_input, past_key_values=past_key_values, use_cache=True)

    N = prefix_ids.shape[1]
    K = draft_tokens.shape[1]

    relevant_logits = out.logits[:, N - 1 : N + K, :]

    if sampling.mode == "greedy":
        target_probs = torch.zeros_like(relevant_logits)
        target_probs.scatter_(-1, relevant_logits.argmax(dim=-1, keepdim=True), 1.0)
    else:
        target_probs = torch.softmax(relevant_logits / sampling.temperature, dim=-1)
    
    return target_probs, past_key_values
