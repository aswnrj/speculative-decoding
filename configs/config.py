import torch
from dataclasses import dataclass

@dataclass
class Config: 
    draft_name: str = 'meta-llama/Llama-3.2-1B'
    target_name: str = 'meta-llama/Llama-3.2-3B'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float16

config = Config()