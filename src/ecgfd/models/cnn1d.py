"""1D CNN on the sampled calibration-pulse response (feature set C3).

Needs the optional dependency: pip install -e ".[dl]"
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PulseCNN(nn.Module):
    def __init__(self, n_classes: int, channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch, n_points]
        z = self.features(x.unsqueeze(1))
        return self.head(z.mean(dim=2))


def _tensor(x: np.ndarray, scale: float) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32) / scale)


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    scale: float = 2.5,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[PulseCNN, list[dict[str, float]]]:
    """Train with Adam and keep the weights of the best validation epoch.

    `y_*` are integer class indices; `scale` normalises volts to about [-1, 1]
    (the ADC full scale), a constant so that no statistic leaks from the data.
    """
    torch.manual_seed(seed)
    model = PulseCNN(n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = _tensor(x_train, scale).to(device), torch.as_tensor(y_train).long().to(device)
    xv, yv = _tensor(x_val, scale).to(device), torch.as_tensor(y_val).long().to(device)
    generator = torch.Generator().manual_seed(seed)

    history: list[dict[str, float]] = []
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(dim=1) == yv).float().mean().item()
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def predict(model: PulseCNN, x: np.ndarray, scale: float = 2.5, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(_tensor(x, scale).to(device)).argmax(dim=1).cpu().numpy()
