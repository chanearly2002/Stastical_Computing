"""TARNet / CFR network (paper Fig. 1, Sec 4 and Sec 5).

Shared representation Phi (n_rep fully-connected ELU layers), normalised to unit L2 norm per sample
(the paper: "normalize Phi through either projection or batch-normalization with fixed scale"; we use
projection onto the unit sphere), followed by two separate heads h0 and h1 (n_head ELU layers + linear output).
"""
import torch
from torch import nn


def _mlp(d_in: int, d_hidden: int, n_layers: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, d_hidden), nn.ELU()]
        d = d_hidden
    return nn.Sequential(*layers)


class CFRNet(nn.Module):
    def __init__(self, d_in: int, d_rep: int = 200, n_rep: int = 3, d_head: int = 100, n_head: int = 3,
                 normalize: bool = True):
        super().__init__()
        self.rep = _mlp(d_in, d_rep, n_rep)
        self.head0 = nn.Sequential(_mlp(d_rep, d_head, n_head), nn.Linear(d_head, 1))
        self.head1 = nn.Sequential(_mlp(d_rep, d_head, n_head), nn.Linear(d_head, 1))
        self.normalize = normalize

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        r = self.rep(x)
        if self.normalize:
            r = r / torch.sqrt((r ** 2).sum(1, keepdim=True) + 1e-12)
        return r

    def forward(self, x: torch.Tensor):
        r = self.representation(x)
        return r, self.head0(r).squeeze(-1), self.head1(r).squeeze(-1)

    def head_weights(self):
        """Weight matrices of the two heads (what the lambda * ||V||^2 penalty of Algorithm 1 acts on)."""
        return [m.weight for h in (self.head0, self.head1) for m in h.modules() if isinstance(m, nn.Linear)]


def sample_weights(t: torch.Tensor) -> torch.Tensor:
    """w_i = t_i / (2u) + (1 - t_i) / (2 (1 - u)),  u = mean(t)   (Eq. 3)."""
    u = t.mean()
    return t / (2 * u) + (1 - t) / (2 * (1 - u))
