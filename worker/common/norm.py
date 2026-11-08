import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Observation normaliser that lives inside the agent, so its stats are saved in state_dict
    (RL spec §4.1 change 5). update() during training only; frozen at eval."""

    def __init__(self, dim, eps=1e-8, clip=10.0, enabled=True):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip, self.enabled = eps, clip, enabled

    @torch.no_grad()
    def update(self, x):  # x: [num_envs, dim]
        if not self.enabled:
            return
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d, tot = bm - self.mean, self.count + bn
        m2 = self.var * self.count + bv * bn + d.pow(2) * self.count * bn / tot
        self.mean.copy_(self.mean + d * bn / tot)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x):
        if not self.enabled:
            return x
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
