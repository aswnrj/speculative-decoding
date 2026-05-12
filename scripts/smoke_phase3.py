import torch
from src.models import load_models
from src.sampling import SamplingConfig
from src.draft import generate_draft


def main():
    bundle = load_models()
    prompt = "The capital of France is"
    input_ids = bundle.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    K = 4
    tokens, probs, _ = generate_draft(
        model=bundle.draft,
        input_ids=input_ids,
        K=K,
        sampling=SamplingConfig(mode="sample", temperature=0.8),
    )

    print(f"tokens shape: {tokens.shape}")
    print(f"probs shape:  {probs.shape}")
    print(f"probs sum:    {probs.sum(-1).tolist()}")
    print(f"decoded:      {bundle.tokenizer.decode(tokens[0])}")


if __name__ == "__main__":
    main()
