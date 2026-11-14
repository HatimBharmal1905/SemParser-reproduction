"""Full experiment grid: ablation study x seeds, with two Hadoop fine-tuning protocols.

  python experiments.py                      # 5 variants x 3 seeds (template split) + full model x 3 seeds (random split)
  python experiments.py --seeds 0 1 2 3 4

Hadoop protocols
  random   : 50 random Hadoop lines for fine-tuning, the other 60 for testing (as in train.py / the paper).
             The same template can appear on both sides, so scores are optimistic.
  template : whole templates are put on one side (>= 50 lines to fine-tune, rest to test), so every test template
             is unseen during fine-tuning. This is the stricter protocol; we use it for the ablation.
Writes results.csv and prints a Markdown summary.
"""
import argparse, copy, csv, os, random, statistics as st
from concurrent.futures import ProcessPoolExecutor

import torch

from data import load, split_by_template, template_key, N_LOC
from model import SemMiner
from train import build_vocabs, run_epochs, evaluate

VARIANTS = [None, 'char', 'local', 'lstm', 'ctx']
NAMES = {None: 'Full model', 'char': 'w/o char CNN', 'local': 'w/o local features',
         'lstm': 'w/o BiLSTM', 'ctx': 'w/o interval context'}


def hadoop_split(had, how, seed, n=50):
    if how == 'random':
        had = list(had); random.shuffle(had)
        return had[:n], had[n:]
    groups = {}
    for s in had:
        groups.setdefault(template_key(s), []).append(s)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    ft, ht = [], []
    for k in keys:
        (ft if len(ft) < n else ht).extend(groups[k])
    return ft, ht


def one(job):
    variant, seed, how, epochs = job
    torch.set_num_threads(1)
    random.seed(seed); torch.manual_seed(seed)
    osd, had = load('openstack_clean.txt'), load('hadoop_fixed.txt')
    words, chars, emb = build_vocabs(osd + had)
    model = SemMiner(len(words), len(chars), N_LOC, emb=emb, ablate=variant)
    tr, te = split_by_template(osd)
    run_epochs(model, tr, words, chars, epochs, 0.01)
    r_os = evaluate(model, te, words, chars)
    ft, ht = hadoop_split(had, how, seed)
    r_before = evaluate(model, ht, words, chars)
    ft_model = copy.deepcopy(model)
    run_epochs(ft_model, ft, words, chars, epochs, 0.01 / 3)
    r_after = evaluate(ft_model, ht, words, chars)
    print(f'done {NAMES[variant]:22s} seed {seed} {how:8s} after-FT F1 {r_after[2]:.3f}', flush=True)
    return dict(variant=NAMES[variant], seed=seed, protocol=how, n_ft=len(ft), n_test=len(ht),
                os_p=r_os[0], os_r=r_os[1], os_f1=r_os[2],
                before_p=r_before[0], before_r=r_before[1], before_f1=r_before[2],
                after_p=r_after[0], after_r=r_after[1], after_f1=r_after[2])


def ms(rows, key):
    xs = [r[key] for r in rows]
    return f'{st.mean(xs):.3f} ± {st.pstdev(xs):.3f}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    jobs = [(v, s, 'template', a.epochs) for v in VARIANTS for s in a.seeds]
    jobs += [(None, s, 'random', a.epochs) for s in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    with open('results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def pick(name, how):
        return [r for r in rows if r['variant'] == name and r['protocol'] == how]

    n = len(a.seeds)
    ex0 = pick('Full model', 'template')[0]
    print(f'\nMean ± std over {n} seeds. Hadoop fine-tune lines / test lines (template protocol, seed 0): '
          f'{ex0["n_ft"]} / {ex0["n_test"]}\n')
    print('| Variant | OpenStack F1 | Hadoop F1 before FT | Hadoop P after FT | Hadoop R after FT | Hadoop F1 after FT |')
    print('|---|---|---|---|---|---|')
    for v in VARIANTS:
        rs = pick(NAMES[v], 'template')
        print(f'| {NAMES[v]} | {ms(rs, "os_f1")} | {ms(rs, "before_f1")} | {ms(rs, "after_p")} | '
              f'{ms(rs, "after_r")} | {ms(rs, "after_f1")} |')
    rs = pick('Full model', 'random')
    print(f'| Full model, random Hadoop split ({rs[0]["n_ft"]}/{rs[0]["n_test"]}) | {ms(rs, "os_f1")} | '
          f'{ms(rs, "before_f1")} | {ms(rs, "after_p")} | {ms(rs, "after_r")} | {ms(rs, "after_f1")} |')
