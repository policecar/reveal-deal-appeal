import torch
import torch.nn as nn

from setfit import SetFitHead


class BottleneckClassifier(SetFitHead):
    def __init__(
        self,
        in_features=None,
        bottleneck_dim=128,
        out_features=2,
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)

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
        return nn.CrossEntropyLoss()
