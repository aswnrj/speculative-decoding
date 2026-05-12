import torch
from transformers import DynamicCache, PreTrainedModel
from typing import Optional
from src.sampling import SamplingConfig, sample_token

@torch.inference_mode()
def generate_draft(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        K: int,
        sampling: SamplingConfig,
        past_key_values: Optional[DynamicCache] = None,
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
    if past_key_values is None:
        past_key_values = DynamicCache()

    draft_tokens: list[torch.Tensor] = []
    draft_probs: list[torch.Tensor] = []

    current = input_ids
    for _ in range(K):
        out = model(input_ids=current, past_key_values=past_key_values, use_cache=True)
        next_logits = out.logits[:, -1, :]

        if sampling.mode == "greedy":
            probs = torch.zeros_like(next_logits)
            probs.scatter_(-1, next_logits.argmax(dim=-1, keepdim=True), 1.0)
        else:
            probs = torch.softmax(next_logits / sampling.temperature, dim=-1)

        next_token = sample_token(next_logits, sampling)
        draft_tokens.append(next_token)
        draft_probs.append(probs)

        current = next_token.unsqueeze(-1)
    
    return (
        torch.stack(draft_tokens, dim=1),
        torch.stack(draft_probs, dim=1),
        past_key_values
    )