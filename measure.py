"""Measure training time, model size and inference speed of this implementation (single CPU thread)."""
import io, random, time

import torch

from data import load, split_by_template, N_LOC
from model import SemMiner
from train import build_vocabs, run_epochs, predict

torch.set_num_threads(1)
random.seed(0); torch.manual_seed(0)
osd, had = load('openstack_clean.txt'), load('hadoop_fixed.txt')
words, chars, emb = build_vocabs(osd + had)
model = SemMiner(len(words), len(chars), N_LOC)
tr, te = split_by_template(osd)

t = time.time(); run_epochs(model, tr, words, chars, 30, 0.01); t_base = time.time() - t
buf = io.BytesIO(); torch.save({'model': model.state_dict()}, buf)
n_par = sum(p.numel() for p in model.parameters())

model.eval(); t = time.time()
for s in had:                      # one message at a time, as in the paper's "single batch" setting
    predict(model, [s], words, chars)
speed = len(had) / (time.time() - t)
print(f'base training (30 epochs, {len(tr)} logs): {t_base:.0f} s')
print(f'parameters: {n_par:,}   checkpoint: {buf.getbuffer().nbytes / 1e6:.1f} MB')
print(f'inference: {speed:.0f} messages/s (batch size 1, 1 thread)')
