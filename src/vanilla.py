import time
import torch
from typing import Optional
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from src.benchmark import measure, reset_peak_vram, BenchmarkResult
from src.sampling import sample_token, SamplingConfig

@torch.inference_mode()
def generate_vanilla(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    sampling: SamplingConfig,
    benchmark: bool = False
) -> tuple[torch.Tensor, str, Optional[BenchmarkResult]]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if benchmark:
        reset_peak_vram()
        start = time.perf_counter()

    past_kv_cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=past_kv_cache, use_cache=True)
    next_token = sample_token(out.logits[:, -1, :], sampling)
    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break
        out = model(
            input_ids=next_token.unsqueeze(-1),
            past_key_values=past_kv_cache,
            use_cache=True
        )
        next_token = sample_token(out.logits[:, -1, :], sampling)
        generated.append(next_token)

    new_ids = torch.stack(generated, dim=1)
    full_ids = torch.cat([input_ids, new_ids], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    bench = measure(num_tokens=new_ids.shape[1], start_time=start) if benchmark else None
    return full_ids, text, bench