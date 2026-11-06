"""Inference-only dependency closure of rynn_transformer_distill_v1/context_v4.
Architecture and forward operations are unchanged; training/data paths omitted.
"""
import torch
from torch import nn

BRANCHES = [('xyz', 0, 3), ('rot', 3, 6), ('grip', 6, 7)]
ARCH = dict(feature_dim=4096, width=128, bottleneck=256, memory_tokens=8,
            layers=2, heads=4, ff_width=256, dropout=.1, classes=256)

class BaseDraft(nn.Module):
    def __init__(self, horizon, dims, architecture=None):
        super().__init__()
        a = architecture or ARCH
        self.horizon, self.dims = horizon, dims
        w = a['width']
        self.condition = nn.Sequential(nn.LayerNorm(a['feature_dim']),
            nn.Linear(a['feature_dim'], a['bottleneck']), nn.GELU(),
            nn.Linear(a['bottleneck'], a['memory_tokens'] * w))
        self.time_queries = nn.Parameter(torch.randn(horizon, w) * .02)
        self.dim_queries = nn.Parameter(torch.randn(dims, w) * .02)
        layer = nn.TransformerDecoderLayer(w, a['heads'], a['ff_width'],
            a['dropout'], activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, a['layers'], norm=nn.LayerNorm(w))
        self.output = nn.Linear(w, a['classes'])
        self.memory_tokens, self.width = a['memory_tokens'], w

class TransformerDraft(BaseDraft):
    def __init__(self, horizon, dims, architecture=None):
        super().__init__(horizon, dims, ARCH)
        del self.condition
        self.context_norm = nn.LayerNorm(4096)
        self.context_projection = nn.Linear(4096, self.width)
        self.proprio = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, self.width),
                                    nn.GELU(), nn.Linear(self.width, self.width))
        self.type_embedding = nn.Parameter(torch.randn(5, self.width) * .02)
        self.gate = nn.Parameter(torch.zeros(self.width))

    def forward(self, x):
        context = x[:, :16384].reshape(-1, 4, 4096)
        memory = self.context_projection(self.context_norm(context))
        memory = torch.cat([memory, self.proprio(x[:, 16384:])[:, None]], 1) + self.type_embedding
        memory = memory * (1 + self.gate.tanh())
        query = (self.time_queries[:, None] + self.dim_queries[None]).reshape(1, -1, self.width).expand(len(x), -1, -1)
        z = self.transformer(query, memory)
        return self.output(z).view(-1, self.horizon, self.dims, 256)

def context_feature(prompt, hidden, state):
    ids = prompt[0]
    starts = (ids == 8197).nonzero().flatten().tolist()
    ends = (ids == 8196).nonzero().flatten().tolist()
    if len(starts) != len(ends) or len(starts) < 2:
        raise ValueError('Need matching current camera/wrist image spans')
    h = hidden[0].float()
    visual = [h[a+1:b].mean(0) for a, b in zip(starts[-2:], ends[-2:])]
    mask = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
    for a, b in zip(starts, ends):
        mask[a:b+1] = False
    text = h[mask].mean(0)
    return torch.cat([h[-1], visual[0], visual[1], text,
                      torch.as_tensor(state, device=h.device, dtype=torch.float32)])
