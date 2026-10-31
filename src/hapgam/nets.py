"""Deep-learning baselines: DNNGP (architecture replicated from the authors' released
Keras model file) and a fully connected MLP on all SNPs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DNNGP(nn.Module):
    """Conv1D(64,4)-Dropout-BN x2, Conv1D(64,4)-BN, Flatten, Dense(3), Dropout, Dense(1)."""

    def __init__(self, n_in: int, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 64, 4, padding="same"), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 4, padding="same"), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 4, padding="same"), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten(), nn.Linear(64 * n_in, 3), nn.Dropout(dropout), nn.Linear(3, 1))

    def forward(self, x):
        return self.body(x[:, None, :]).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 64), dropout: float = 0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x).squeeze(-1)


def train_net(make, Xtr, ytr, Xva, yva, lr=1e-3, wd=0.0, batch=256, max_epochs=300, patience=20,
              seed=0, device="cuda", fixed_epochs=None):
    """Adam + MSE with early stopping on (Xva, yva); if fixed_epochs is given, train that long
    without validation.  Returns (model, best_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(patience // 4, 2))
    ym, ys = float(ytr.mean()), float(ytr.std())
    yt = (ytr - ym) / ys
    n = len(yt)
    best, best_state, best_ep, bad = -np.inf, None, 0, 0
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = ((net(Xtr[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        if fixed_epochs is not None:
            continue
        net.eval()
        with torch.no_grad():
            pv = net(Xva).cpu().numpy()
        r = float(np.corrcoef(pv, yva.cpu().numpy())[0, 1]) if np.std(pv) > 0 else -1.0
        sched.step(-r)
        if r > best + 1e-4:
            best, best_ep, bad = r, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net.y_mean, net.y_sd = ym, ys
    return net, (best_ep if fixed_epochs is None else fixed_epochs), best


@torch.no_grad()
def predict_net(net, X):
    net.eval()
    return net(X).cpu().numpy() * net.y_sd + net.y_mean
