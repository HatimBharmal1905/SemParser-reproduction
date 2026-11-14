"""Parse the annotated log files (C:word, I:word@@concept, I:word) into training samples."""
import random

TMP = '<TMP>'
NONE, CONCEPT, INSTANCE = 0, 1, 2


def parse_line(line):
    """Returns dict(tokens, types, pairs). pairs = [(concept_idx, instance_idx)] (token indices, no <TMP>)."""
    tokens, types, inst_concept = [], [], {}
    for raw in line.strip().split(' '):
        if raw.startswith('C:'):
            tokens.append(raw[2:]); types.append(CONCEPT)
        elif raw.startswith('I:'):
            w = raw[2:]
            if '@@' in w:
                w, c = w.rsplit('@@', 1)
                inst_concept[len(tokens)] = c
            tokens.append(w); types.append(INSTANCE)
        else:
            tokens.append(raw); types.append(NONE)
    pairs = []
    for i, c in inst_concept.items():
        js = [j for j in range(i) if types[j] == CONCEPT and tokens[j] == c]
        if js:
            pairs.append((js[-1], i))  # nearest preceding concept with the same word
    return {'tokens': tokens, 'types': types, 'pairs': pairs}


def load(path):
    with open(path, encoding='utf-8') as f:
        return [parse_line(l) for l in f if l.strip() and not l.startswith('#')]


def template_key(s):
    return tuple('<I>' if t == INSTANCE else w for w, t in zip(s['tokens'], s['types']))


def split_by_template(samples, train_ratio=0.8, seed=0):
    """Split so that test templates are unseen (avoids leakage between near-duplicate lines)."""
    groups = {}
    for s in samples:
        groups.setdefault(template_key(s), []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * train_ratio)
    tr = [s for k in keys[:cut] for s in groups[k]]
    te = [s for k in keys[cut:] for s in groups[k]]
    return tr, te


def stats(samples):
    n_c = sum(t == CONCEPT for s in samples for t in s['types'])
    n_i = sum(t == INSTANCE for s in samples for t in s['types'])
    n_p = sum(len(s['pairs']) for s in samples)
    return {'lines': len(samples),
            'tokens': sum(len(s['tokens']) for s in samples),
            'concept_tokens': n_c,
            'unique_concepts': len({w for s in samples for w, t in zip(s['tokens'], s['types']) if t == CONCEPT}),
            'instance_tokens': n_i, 'ci_pairs': n_p, 'orphan_instances': n_i - n_p,
            'templates': len({template_key(s) for s in samples})}


def loc_feats(t):
    d = sum(ch.isdigit() for ch in t)
    return [min(len(t), 30) / 30, d / max(len(t), 1), float(d > 0), float(any(ch.isupper() for ch in t)),
            float(t[:1].isupper()), float(any(ch in '/\\' for ch in t)), float('.' in t),
            float(any(ch in '-_:' for ch in t))]


N_LOC = 8

if __name__ == '__main__':
    for f in ['openstack_clean.txt', 'hadoop_fixed.txt']:
        print(f, stats(load(f)))
