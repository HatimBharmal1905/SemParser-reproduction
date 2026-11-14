import torch
import torch.nn as nn
import torch.nn.functional as F


class SemMiner(nn.Module):
    """Contextual encoder (word emb + char CNN + local feats -> 2-layer BiLSTM), pair matcher, word scorer."""

    def __init__(self, n_words, n_chars, n_loc, wdim=100, cdim=30, cfilt=50, hid=128, layers=2, drop=0.3, emb=None,
                 ablate=None):
        super().__init__()
        assert ablate in (None, 'char', 'local', 'lstm', 'ctx')
        self.ablate = ablate  # ablation switch: which component is removed (None = full model)
        self.wemb = nn.Embedding(n_words, wdim, padding_idx=0)
        if emb is not None:
            self.wemb.weight.data.copy_(emb)
        self.cemb = nn.Embedding(n_chars, cdim, padding_idx=0)
        self.ccnn = nn.Conv1d(cdim, cfilt, 3, padding=1)
        D = 2 * hid
        if ablate == 'lstm':   # "w/o LSTM": per-word linear layer, no sequence context
            self.lin = nn.Sequential(nn.Linear(wdim + cfilt + n_loc, D), nn.Tanh())
        else:
            self.lstm = nn.LSTM(wdim + cfilt + n_loc, hid, num_layers=layers, bidirectional=True,
                                batch_first=True, dropout=drop)
        self.pair = nn.Sequential(nn.Linear(2 * D + wdim, hid), nn.ReLU(), nn.Linear(hid, 1))   # FFNN_a
        self.word = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, 3))              # FFNN_b
        self.drop = nn.Dropout(drop)

    def forward(self, w, c, f):
        B, T = w.shape
        e = self.wemb(w)                                                    # B,T,wdim
        ch = self.cemb(c.view(B * T, -1)).transpose(1, 2)                   # B*T,cdim,L
        ch = F.relu(self.ccnn(ch)).max(dim=2).values.view(B, T, -1)         # B,T,cfilt
        if self.ablate == 'char':
            ch = torch.zeros_like(ch)                                       # w/o F_char
        if self.ablate == 'local':
            f = torch.zeros_like(f)                                         # w/o F_local
        x = self.drop(torch.cat([e, ch, f], -1))
        m = self.lin(x) if self.ablate == 'lstm' else self.lstm(x)[0]       # B,T,D
        D = m.size(-1)
        # interval context: mean word embedding strictly between j and i
        cs = torch.cat([torch.zeros(B, 1, e.size(2), device=e.device), e.cumsum(1)], 1)
        idx = torch.arange(T, device=w.device)
        ii, jj = idx.view(1, T, 1), idx.view(1, 1, T)
        s = cs[:, :T].unsqueeze(2) - cs[:, 1:T + 1].unsqueeze(1)            # B,T,T,wdim
        cnt = (ii - jj - 1).clamp(min=0).float().unsqueeze(-1)              # 1,T,T,1
        contx = s / cnt.clamp(min=1) * (cnt > 0).float()
        if self.ablate == 'ctx':
            contx = torch.zeros_like(contx)                                 # w/o F_contx
        feat = torch.cat([m.unsqueeze(2).expand(B, T, T, D), m.unsqueeze(1).expand(B, T, T, D), contx], -1)
        pscore = self.pair(feat).squeeze(-1).masked_fill(~(jj < ii), -1e9)  # B,T,T  (only j < i)
        return pscore, self.word(m)                                         # B,T,3


def loss_fn(pscore, wscore, pt, wt):
    T = pscore.size(-1)
    return (F.cross_entropy(pscore.reshape(-1, T), pt.reshape(-1), ignore_index=-100) +
            F.cross_entropy(wscore.reshape(-1, 3), wt.reshape(-1), ignore_index=-100))
