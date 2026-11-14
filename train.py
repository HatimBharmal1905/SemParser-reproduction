"""Stage 1: train base model on OpenStack. Stage 2: fine-tune on N Hadoop lines, test on the rest."""
import argparse, copy, random
import torch
from data import load, split_by_template, loc_feats, N_LOC, TMP, INSTANCE, CONCEPT
from model import SemMiner, loss_fn
from joint_parser import DomainKnowledge, joint_parse

MAXC = 20


def build_vocabs(samples, emb_path=None):
    words = {'<PAD>': 0, '<UNK>': 1, TMP: 2}
    chars = {'<PAD>': 0, '<UNK>': 1}
    for s in samples:
        for w in s['tokens']:
            words.setdefault(w.lower(), len(words))
            for ch in w[:MAXC]:
                chars.setdefault(ch, len(chars))
    emb = None
    if emb_path:                                  # optional gensim skip-gram vectors (dim 100)
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(emb_path)
        emb = torch.randn(len(words), kv.vector_size) * 0.1
        for w, i in words.items():
            if w in kv:
                emb[i] = torch.tensor(kv[w])
    return words, chars, emb


def batchify(samples, words, chars):
    T = max(len(s['tokens']) for s in samples) + 1
    B = len(samples)
    w = torch.zeros(B, T, dtype=torch.long); c = torch.zeros(B, T, MAXC, dtype=torch.long)
    f = torch.zeros(B, T, N_LOC); wt = torch.full((B, T), -100); pt = torch.full((B, T), -100)
    for b, s in enumerate(samples):
        toks = [TMP] + s['tokens']
        for k, t in enumerate(toks):
            w[b, k] = words.get(t.lower(), 1)
            for q, ch in enumerate(t[:MAXC]):
                c[b, k, q] = chars.get(ch, 1)
            f[b, k] = torch.tensor(loc_feats(t) if k else [0.0] * N_LOC)
        for k, ty in enumerate(s['types']):
            wt[b, k + 1] = ty; pt[b, k + 1] = 0           # default target: <TMP>
        for j, i in s['pairs']:
            pt[b, i + 1] = j + 1
    return w, c, f, pt, wt


def run_epochs(model, data, words, chars, epochs, lr, bs=8):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for g in opt.param_groups:
            g['lr'] = lr / (1 + 0.05 * ep)
        random.shuffle(data); model.train(); tot = 0
        for k in range(0, len(data), bs):
            w, c, f, pt, wt = batchify(data[k:k + bs], words, chars)
            ps, ws = model(w, c, f)
            loss = loss_fn(ps, ws, pt, wt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  epoch {ep + 1}/{epochs} loss {tot:.3f}')


@torch.no_grad()
def predict(model, samples, words, chars):
    model.eval()
    w, c, f, _, _ = batchify(samples, words, chars)
    ps, ws = model(w, c, f)
    out = []
    for b, s in enumerate(samples):
        n = len(s['tokens'])
        types = ws[b, 1:n + 1].argmax(-1).tolist()
        pairs = []
        for i in range(1, n + 1):
            j = ps[b, i, :i].argmax().item()
            if j > 0 and types[i - 1] == INSTANCE:
                pairs.append((j - 1, i - 1))
        out.append((types, pairs))
    return out


def evaluate(model, samples, words, chars):
    tp = fp = fn = 0
    for s, (_, pred) in zip(samples, predict(model, samples, words, chars)):
        g, p = set(s['pairs']), set(pred)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--openstack', default='openstack_clean.txt'); ap.add_argument('--hadoop', default='hadoop_fixed.txt')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--ft_n', type=int, default=50); ap.add_argument('--emb', default=None)
    ap.add_argument('--ablate', default=None, choices=['char', 'local', 'lstm', 'ctx'],
                    help='remove one component (ablation study); omit for the full model')
    a = ap.parse_args(); random.seed(0); torch.manual_seed(0)
    osd, had = load(a.openstack), load(a.hadoop)
    words, chars, emb = build_vocabs(osd + had, a.emb)
    model = SemMiner(len(words), len(chars), N_LOC, emb=emb, ablate=a.ablate)

    tr, te = split_by_template(osd)
    print('Stage 1: base model on OpenStack', len(tr), 'train /', len(te), 'test (unseen templates)')
    run_epochs(model, tr, words, chars, a.epochs, a.lr)
    print('OpenStack  P/R/F1: %.3f %.3f %.3f' % evaluate(model, te, words, chars))

    random.shuffle(had); ft, ht = had[:a.ft_n], had[a.ft_n:]
    print('Hadoop before fine-tune P/R/F1: %.3f %.3f %.3f' % evaluate(model, ht, words, chars))
    ft_model = copy.deepcopy(model)
    run_epochs(ft_model, ft, words, chars, a.epochs, a.lr / 3)
    print('Hadoop after  fine-tune P/R/F1: %.3f %.3f %.3f' % evaluate(ft_model, ht, words, chars))
    torch.save({'model': ft_model.state_dict(), 'words': words, 'chars': chars}, 'semparser.pt')

    dk = DomainKnowledge()                              # joint parser demo over the Hadoop test lines
    for s, (types, pairs) in list(zip(ht, predict(ft_model, ht, words, chars)))[:5]:
        print(joint_parse(s['tokens'], types, pairs, dk))
