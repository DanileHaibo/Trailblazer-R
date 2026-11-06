"""Original rich components: independent branches, no action-tail input.

Canonical input uses previous/current agentview (single image at first query);
when only one image exists its real token pools are duplicated. No wrist input.
The checkpoint's legacy config memory label mentions wrist, but canonical_v2
collected its features without wrist. Architecture and weights are unchanged.
"""
import torch
from torch import nn

def context(pref, prompt, obs):
    h = pref.hidden_states[-1][0].float(); ids = prompt[0].tolist(); spans = []; start = None
    mask = torch.zeros(len(ids), dtype=torch.bool, device=h.device)
    for i, t in enumerate(ids):
        if t == 8197:
            start = i
        elif t == 8196 and start is not None:
            spans.append((start, i+1)); mask[start:i+1] = True; start = None
    memory = [h[-1], h[~mask].mean(0) if (~mask).any() else h[-1]]
    for a, b in (spans[-2:] if len(spans) >= 2 else spans*2):
        memory.extend([z.mean(0) for z in torch.tensor_split(h[a:b], 4)])
    if len(memory) != 10:
        raise ValueError('one or two canonical agentview image-token spans required')
    from libero_util.libero_utils import quat2axisangle
    import numpy as np
    state = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']])
    return torch.stack(memory), torch.as_tensor(state, device=h.device, dtype=h.dtype)

class Branch(nn.Module):
    def __init__(self, horizon, dims, hidden=128):
        super().__init__(); self.horizon = horizon; self.dims = dims
        self.proj = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, hidden))
        self.state = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, hidden))
        self.modality = nn.Parameter(torch.randn(11, hidden)*.02)
        self.time = nn.Parameter(torch.randn(horizon, hidden)*.02)
        self.dim = nn.Parameter(torch.randn(dims, hidden)*.02)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden, 4, hidden*4, dropout=.1, batch_first=True, norm_first=True), 2, norm=nn.LayerNorm(hidden))
        self.gate = nn.Parameter(torch.tensor(-1.)); self.scalar = nn.Linear(hidden, 257); self.boundary = nn.Linear(hidden, 3)
    def forward(self, memory, state):
        m = torch.cat([self.proj(memory), self.state(state)[:, None]], 1)+self.modality
        q = (self.time[:, None]+self.dim[None, :]).reshape(1, self.horizon*self.dims, -1).expand(len(memory), -1, -1)
        z = q+self.gate.sigmoid()*self.decoder(q, m)
        return self.scalar(z).reshape(len(memory), self.horizon, self.dims, 257), self.boundary(z.reshape(len(memory), self.horizon, self.dims, -1).mean(2))

class RichDraft(nn.Module):
    def __init__(self, horizon):
        super().__init__(); self.horizon = horizon
        self.heads = nn.ModuleList([Branch(horizon, d) for d in (3, 3, 1)])
    def forward(self, memory, state):
        return [head(memory, state) for head in self.heads]
    @torch.inference_mode()
    def propose(self, features):
        memory, state = features
        outputs = self(memory[None], state[None])
        ids = torch.cat([x[0].argmax(-1) for x in outputs], 2)[0]+10005
        return sum(([10004]+row+[15004] for row in ids.tolist()), [])+[8710]
