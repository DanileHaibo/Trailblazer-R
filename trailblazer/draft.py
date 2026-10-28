"""Small temporal draft; official target weights are never included."""
import numpy as np
import torch
from torch import nn


def temporal(state, previous, N, M):
    if previous is None:
        return np.r_[state, np.zeros(N * 7)].astype(np.float32), np.zeros(2 + 2 * N, np.float32)
    tail = np.asarray(previous, np.float32)[M:]
    if tail.shape != (N - M, 7):
        raise ValueError('previous chunk has wrong shape')
    padded = np.concatenate([tail, np.repeat(tail[-1:], M, axis=0)])
    mask = np.r_[1, M, np.ones(N-M), np.zeros(M), np.r_[np.arange(M, N), np.repeat(N-1, M)] / N]
    return np.r_[state, padded.flatten()].astype(np.float32), mask.astype(np.float32)


class GatedDraft(nn.Module):
    def __init__(self, N, memory_dim, width=384, layers=3, dropout=.1):
        super().__init__()
        self.N = N
        self.memory_proj = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, width))
        self.state_proj = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, width), nn.GELU())
        self.history_proj = nn.Sequential(nn.LayerNorm(N*7), nn.Linear(N*7, width), nn.GELU())
        self.tail_mask_proj = nn.Sequential(nn.Linear(2+2*N, width), nn.GELU())
        self.action_query = nn.Parameter(torch.randn(1, 1, width)*.02)
        self.position = nn.Embedding(N, width)
        layer = nn.TransformerDecoderLayer(width, 4, width*4, dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        self.action_head = nn.Linear(width, 7)
        self.gate_head = nn.Linear(width, 6)
        self.change_head = nn.Linear(width, 1)

    def forward(self, memory, prior, padding, mask):
        c = self.state_proj(prior[:, :8].float()) + self.history_proj(prior[:, 8:].float()) + self.tail_mask_proj(mask.float())
        h = self.decoder(self.action_query + self.position.weight[None] + c[:, None], self.memory_proj(memory.float()), memory_key_padding_mask=padding)
        delta = self.action_head(h)
        base = prior[:, 8:].reshape(-1, self.N, 7)
        continuous = base[..., :6] + torch.sigmoid(self.gate_head(h))*delta[..., :6]
        grip = base[..., 6:] + (torch.sigmoid(self.change_head(h)) > .9).to(delta.dtype)*delta[..., 6:]
        return torch.cat([continuous, grip], -1)


def load_draft(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['kind'] != 'gated':
        raise ValueError('Only released gated drafts are supported')
    model = GatedDraft(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint
