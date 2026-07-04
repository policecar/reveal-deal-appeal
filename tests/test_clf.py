"""Unit tests for BottleneckClassifier construction and forward pass."""

import torch

from clf import BottleneckClassifier


def test_head_constructs_with_explicit_in_features():
    """SetFitHead xavier-inits all nn.Linear at __init__; a LazyLinear
    (in_features=None) crashes there, so explicit in_features must work."""
    head = BottleneckClassifier(
        in_features=768, bottleneck_dim=128, out_features=2, device="cpu"
    )
    logits, probs = head(torch.randn(4, 768))

    assert logits.shape == (4, 2)
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4))
