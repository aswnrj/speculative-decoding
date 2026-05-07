import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from configs.config import config

@dataclass
class ModuleBundle:
    draft: torch.nn.Module
    target: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    vram_after_draft_gb: float
    vram_after_target_gb: float

def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def _load_model(name) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=config.dtype,
        attn_implementation="sdpa",
    ).to(config.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_models() -> ModuleBundle:
    tokenizer = AutoTokenizer.from_pretrained(config.target_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading draft: {config.draft_name}")
    draft = _load_model(config.draft_name)
    vram_draft = _vram_gb()
    print(f"VRAM after draft: {vram_draft:.2f} GB")

    print(f"Loading target: {config.target_name}")
    target = _load_model(config.target_name)
    vram_target = _vram_gb()
    print(f"VRAM after target: {vram_target:.2f} GB")

    return ModuleBundle(
        draft=draft,
        target=target,
        tokenizer=tokenizer,
        vram_after_draft_gb=vram_draft,
        vram_after_target_gb=vram_target,
    )

if __name__ == "__main__":
    bundle = load_models()
    print(f"Total VRAM used: {bundle.vram_after_target_gb:.2f} GB")