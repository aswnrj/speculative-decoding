import torch
from src.sampling import SamplingConfig

@torch.inference_mode()
def accept_and_resample(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    sampling: SamplingConfig,
) -> tuple[torch.Tensor, int]:
    K = draft_tokens.shape[1]
    emitted: list[torch.Tensor] = []
    num_accepted = 0

    for i in range(K):
        x = draft_tokens[0, i].item()
        p_t = target_probs[0, i, x].item()
        p_d = draft_probs[0, i, x].item()

        accept_prob = 1.0 if p_d <= 0 else min(1.0, p_t / p_d)
        r = torch.rand(1).item()

        if r < accept_prob:
            emitted.append(draft_tokens[0, i])
            num_accepted += 1
            continue

        residual = torch.clamp(target_probs[0, i, :] - draft_probs[0, i, :], min=0)
        total = residual.sum()
        if total > 0:
            residual = residual / total
        else:
            residual = target_probs[0, i, :]
        resampled = torch.multinomial(residual, num_samples=1).squeeze(0)
        emitted.append(resampled)
        break

    if num_accepted == K:
        bonus_dist = target_probs[0, K, :]
        bonus = torch.multinomial(bonus_dist, num_samples=1).squeeze(0)
        emitted.append(bonus)
        
    emitted_tensor = torch.stack(emitted).unsqueeze(0)
    return emitted_tensor, num_accepted