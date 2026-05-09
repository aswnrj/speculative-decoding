import torch

from src.sampling import SamplingConfig, sample_token


def test_greedy_returns_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    out = sample_token(logits, SamplingConfig(mode="greedy"))
    assert out.shape == (1,)
    assert out.item() == 1


def test_greedy_batch():
    logits = torch.tensor([[1.0, 5.0], [3.0, 0.5]])
    out = sample_token(logits, SamplingConfig(mode="greedy"))
    assert out.shape == (2,)
    assert out.tolist() == [1, 0]


def test_sample_extreme_logits_picks_dominant():
    logits = torch.tensor([[100.0, 0.0, 0.0]])
    torch.manual_seed(0)
    out = sample_token(logits, SamplingConfig(mode="sample", temperature=1.0))
    assert out.item() == 0


def test_sample_returns_valid_token():
    torch.manual_seed(42)
    logits = torch.randn(1, 32)
    out = sample_token(logits, SamplingConfig(mode="sample", temperature=0.8))
    assert 0 <= out.item() < 32
