import torch
import torch.nn as nn
import torch.nn.functional as F

from setfit import SetFitHead


class FocalLoss(nn.Module):
    """Multiclass focal loss: cross-entropy scaled by (1 - pt)^gamma, so
    well-classified (easy, majority) examples contribute little and training
    focuses on the hard minority. Optionally combined with class weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class BottleneckClassifier(SetFitHead):
    def __init__(
        self,
        in_features=None,
        bottleneck_dim=128,
        out_features=2,
        dropout_rate=0.2,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float | None = None,
        **kwargs,
    ):
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)

        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

        self.bottleneck = nn.Sequential(
            nn.LazyLinear(bottleneck_dim)
            if in_features is None
            else nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, self.linear.out_features),
        ).to(self.device)

    def forward(self, features, temperature=None):
        temperature = temperature or self.temperature
        x = features["sentence_embedding"] if isinstance(features, dict) else features

        logits = self.bottleneck(x)
        logits = logits / (temperature + self.eps)
        probs = nn.functional.softmax(logits, dim=-1)

        if isinstance(features, dict):
            features.update({"logits": logits, "probs": probs})
            return features

        return logits, probs

    def predict_proba(self, x_test: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x_test)[1]

    def predict(self, x_test: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x_test)
        return torch.argmax(probs, dim=-1)

    def get_loss_fn(self) -> nn.Module:
        weight = (
            self.class_weights.to(self.device)
            if self.class_weights is not None
            else None
        )
        if self.focal_gamma is not None:
            return FocalLoss(gamma=self.focal_gamma, weight=weight)
        return nn.CrossEntropyLoss(weight=weight)
