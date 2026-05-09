from dataclasses import dataclass
from typing import Literal
import torch

@dataclass
class SamplingConfig:
    mode: Literal["greedy", "sample"] = "greedy"
    temperature: float = 1.0

def sample_token(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    if cfg.mode == "greedy":
        return logits.argmax(dim=-1)
    
    probs = torch.softmax(logits/cfg.temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)