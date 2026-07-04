"""Unit tests for BottleneckClassifier construction, forward pass, and losses."""

import torch
import torch.nn as nn

from clf import BottleneckClassifier, FocalLoss


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


def test_loss_fn_selection():
    """class_weights -> weighted CE; focal_gamma -> FocalLoss (weighted)."""
    weights = torch.tensor([0.52, 13.4])

    head = BottleneckClassifier(in_features=8, class_weights=weights, device="cpu")
    loss = head.get_loss_fn()
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert torch.allclose(loss.weight, weights)

    head = BottleneckClassifier(
        in_features=8, class_weights=weights, focal_gamma=2.0, device="cpu"
    )
    loss = head.get_loss_fn()
    assert isinstance(loss, FocalLoss)
    assert torch.allclose(loss.weight, weights)

    # unweighted default still works
    head = BottleneckClassifier(in_features=8, device="cpu")
    assert head.get_loss_fn().weight is None


def test_focal_loss_downweights_easy_examples():
    """A confident correct prediction should contribute ~nothing to focal
    loss while still incurring plain CE loss."""
    focal = FocalLoss(gamma=2.0)
    ce = nn.CrossEntropyLoss()

    easy_logits = torch.tensor([[8.0, -8.0]])
    target = torch.tensor([0])

    assert focal(easy_logits, target) < 1e-6
    assert focal(easy_logits, target) < ce(easy_logits, target)

    hard_logits = torch.tensor([[-2.0, 2.0]])  # confidently wrong
    assert focal(hard_logits, target) > 1.0
