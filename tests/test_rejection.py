import torch

from src.rejection import accept_and_resample
from src.sampling import SamplingConfig


def _onehot(vocab: int, idx: int) -> torch.Tensor:
    v = torch.zeros(vocab)
    v[idx] = 1.0
    return v


def test_all_accept_emits_K_plus_1():
    """Greedy with target == draft → all K accept, bonus emits → K+1 tokens."""
    K, vocab = 4, 8
    draft_tokens = torch.tensor([[1, 2, 3, 4]])
    draft_probs = torch.zeros(1, K, vocab)
    target_probs = torch.zeros(1, K + 1, vocab)
    for i, x in enumerate(draft_tokens[0]):
        draft_probs[0, i] = _onehot(vocab, x.item())
        target_probs[0, i] = _onehot(vocab, x.item())
    target_probs[0, K] = _onehot(vocab, 5)  # bonus

    emitted, n = accept_and_resample(draft_tokens, draft_probs, target_probs, SamplingConfig(mode="greedy"))
    assert n == K
    assert emitted.shape == (1, K + 1)
    assert emitted[0].tolist() == [1, 2, 3, 4, 5]


def test_reject_at_position_0_emits_residual():
    """Greedy with disagreement at position 0 → 0 accepts, 1 token emitted from residual."""
    K, vocab = 3, 5
    draft_tokens = torch.tensor([[0, 1, 2]])
    draft_probs = torch.zeros(1, K, vocab)
    target_probs = torch.zeros(1, K + 1, vocab)
    for i, x in enumerate(draft_tokens[0]):
        draft_probs[0, i] = _onehot(vocab, x.item())
    target_probs[0, 0] = _onehot(vocab, 3)  # disagrees with draft's 0
    target_probs[0, 1] = _onehot(vocab, 1)
    target_probs[0, 2] = _onehot(vocab, 2)
    target_probs[0, 3] = _onehot(vocab, 4)  # bonus — should NOT be emitted

    emitted, n = accept_and_resample(draft_tokens, draft_probs, target_probs, SamplingConfig(mode="greedy"))
    assert n == 0
    assert emitted.shape == (1, 1)
    assert emitted[0, 0].item() == 3


def test_reject_at_position_1_emits_two_tokens():
    """Accept d1, reject d2 → emit d1 + residual at position 1; d3 discarded."""
    K, vocab = 3, 5
    draft_tokens = torch.tensor([[0, 1, 2]])
    draft_probs = torch.zeros(1, K, vocab)
    target_probs = torch.zeros(1, K + 1, vocab)
    for i, x in enumerate(draft_tokens[0]):
        draft_probs[0, i] = _onehot(vocab, x.item())
    target_probs[0, 0] = _onehot(vocab, 0)  # accept d1
    target_probs[0, 1] = _onehot(vocab, 3)  # reject d2, residual = 3
    target_probs[0, 2] = _onehot(vocab, 2)  # would be d3, ignored
    target_probs[0, 3] = _onehot(vocab, 4)  # bonus, ignored

    emitted, n = accept_and_resample(draft_tokens, draft_probs, target_probs, SamplingConfig(mode="greedy"))
    assert n == 1
    assert emitted.shape == (1, 2)
    assert emitted[0].tolist() == [0, 3]


def test_stochastic_acceptance_probability():
    """Statistical check: when p_t/p_d = 0.5, acceptance rate ≈ 50% over many trials."""
    K, vocab = 1, 4
    draft_tokens = torch.tensor([[0]])
    draft_probs = torch.zeros(1, K, vocab)
    target_probs = torch.zeros(1, K + 1, vocab)
    draft_probs[0, 0] = torch.tensor([0.6, 0.4, 0.0, 0.0])
    target_probs[0, 0] = torch.tensor([0.3, 0.3, 0.2, 0.2])  # p_t/p_d for token 0 = 0.5
    target_probs[0, 1] = _onehot(vocab, 0)  # bonus

    torch.manual_seed(0)
    accepts = 0
    trials = 2000
    for _ in range(trials):
        _, n = accept_and_resample(draft_tokens, draft_probs, target_probs, SamplingConfig(mode="sample", temperature=1.0))
        accepts += (n == 1)
    rate = accepts / trials
    assert 0.45 < rate < 0.55, f"expected ~0.50, got {rate}"
