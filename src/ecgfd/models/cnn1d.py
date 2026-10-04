"""1D CNN on the sampled calibration-pulse response (feature set C3).

Needs the optional dependency: pip install -e ".[dl]"
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PulseCNN(nn.Module):
    """Five convolution blocks that halve the length, then a dense head.

    The feature map is flattened, not averaged over time: what identifies a fault is
    where the response departs from the nominal one (edge, droop, undershoot, tail),
    and a global average would throw that position away.
    """

    def __init__(
        self,
        n_classes: int,
        n_points: int = 1000,
        channels: tuple[int, ...] = (16, 32, 32, 64, 64),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in, length = 1, n_points
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in, length = c_out, length // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c_in * length, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch, n_points]
        return self.head(self.features(x.unsqueeze(1)))


def default_device() -> str:
    """Apple GPU or CUDA when present (about 25 times faster than CPU here), else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tensor(x: np.ndarray, offset: float, scale: float) -> torch.Tensor:
    return torch.as_tensor((np.asarray(x, dtype=np.float32) - offset) / scale)


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-3,
    offset: float = 0.0,
    scale: float = 1.0,
    seed: int = 0,
    device: str | None = None,
) -> tuple[PulseCNN, list[dict[str, float]]]:
    """Train with AdamW and a cosine schedule; keep the best validation epoch.

    `y_*` are integer class indices. `offset` and `scale` map volts to about [-1, 1]:
    use the ADC mid-scale and half range, constants of the instrument, so that no
    statistic leaks from the data.
    """
    device = device or default_device()
    torch.manual_seed(seed)
    model = PulseCNN(n_classes, n_points=x_train.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    xt = _tensor(x_train, offset, scale).to(device)
    xv = _tensor(x_val, offset, scale).to(device)
    yt = torch.as_tensor(y_train).long().to(device)
    yv = torch.as_tensor(y_val).long().to(device)
    generator = torch.Generator().manual_seed(seed)

    history: list[dict[str, float]] = []
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)  # drawn on CPU: same on any device
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(dim=1) == yv).float().mean().item()
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def predict(model: PulseCNN, x: np.ndarray, offset: float = 0.0, scale: float = 1.0) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(_tensor(x, offset, scale).to(device)).argmax(dim=1).cpu().numpy()
