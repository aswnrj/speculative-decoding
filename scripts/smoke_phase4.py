import torch

from src.draft import generate_draft
from src.models import load_models
from src.sampling import SamplingConfig
from src.verify import verify_target


def main():
    bundle = load_models()
    tokenizer = bundle.tokenizer

    prompt = "The capital of France is"
    prefix_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    K = 4
    sampling = SamplingConfig(mode="sample", temperature=0.8)

    draft_tokens, draft_probs, _ = generate_draft(
        model=bundle.draft,
        input_ids=prefix_ids,
        K=K,
        sampling=sampling,
    )

    target_probs, _ = verify_target(
        model=bundle.target,
        prefix_ids=prefix_ids,
        draft_tokens=draft_tokens,
        sampling=sampling,
    )

    print(f"prefix length:       {prefix_ids.shape[1]}")
    print(f"draft tokens:        {draft_tokens.shape}")
    print(f"draft probs:         {draft_probs.shape}")
    print(f"target probs:        {target_probs.shape}  (should be (1, K+1, vocab) = (1, {K+1}, vocab))")
    print(f"target probs sum:    {target_probs.sum(-1).tolist()}")

    target_argmax = target_probs.argmax(dim=-1)[0]  
    draft_seq = draft_tokens[0]                     
    print(f"draft proposed:      {tokenizer.decode(draft_seq)}")
    print(f"target argmax (K+1): {tokenizer.decode(target_argmax)}")


if __name__ == "__main__":
    main()
