# SemParser: reproduction of "SemParser: A Semantic Parser for Log Analytics"

Re-implementation of the semantics miner and joint parser from Huo, Su, Lee and Lyu (arXiv:2112.12636), in PyTorch on CPU.
The miner tags each log token as concept / instance / none and links each instance to its concept (a CI pair). The joint
parser then uses a domain knowledge database to recover implicit CI pairs and builds conceptualised templates.

## Files

| File | Purpose |
|---|---|
| `data.py` | Parse annotations (`C:word`, `I:word@@concept`, `I:word`), template-disjoint split, local features |
| `model.py` | Word embedding + char CNN + local features, 2-layer BiLSTM, pair matcher, word scorer (`ablate` option) |
| `train.py` | Stage 1: train on OpenStack. Stage 2: fine-tune on 50 Hadoop lines, test on the rest |
| `joint_parser.py` | Domain knowledge database, implicit CI pairs, conceptualised templates |
| `experiments.py` | Ablation grid (5 variants x seeds) and both Hadoop protocols; writes `results.csv` |
| `measure.py` | Training time, model size and inference speed (1 CPU thread) |
| `openstack_clean.txt`, `hadoop_fixed.txt` | Annotated logs (115 and 110 messages) from Loghub |
| `results.csv`, `experiments.log` | Raw results of the last `experiments.py` run |
| `report/` | Project report (LaTeX + PDF) and architecture diagram |

## Run

```
pip install torch            # gensim only if you pass --emb
python train.py                         # full model
python train.py --ablate ctx            # char | local | lstm | ctx
python experiments.py                   # 18 runs, 2 worker processes
python measure.py
python joint_parser.py                  # joint parser demo
```

## Annotation format

One message per line, tokens separated by spaces. `C:cell` marks a concept, `I:949e1227@@cell` an instance with its concept,
`I:nova.context()` an orphan instance (concept not in the message). A concept must come before its instance.
Templates are derived from the annotations, and the split keeps whole templates on one side.

## Results (mean of 3 seeds, CI-pair F1)

| Setting | F1 |
|---|---|
| OpenStack, unseen templates | 0.96 |
| Hadoop before fine-tuning (template split) | 0.55 |
| Hadoop after fine-tuning on 50 messages (template split) | 0.82 |
| Hadoop after fine-tuning on 50 messages (random split) | 0.93 (optimistic: templates leak) |

Ablation (Hadoop, template split, delta F1 against the full model): w/o char CNN -0.07, w/o local features +0.01,
w/o BiLSTM -0.03, w/o interval context -0.05. The paper's large drop without interval context is **not** reproduced.
Test sets have only 26 to 60 messages and the std over seeds is up to 0.10, so read these as trends.
Our numbers are lower than the paper's 0.985 mainly because of the much smaller data (about 110 messages per system,
randomly initialised embeddings) and the stricter template-disjoint protocol.

## Limitations

- Annotations were produced by heuristic pre-labelling followed by manual clean-up, with a single annotator and no
  inter-annotator agreement.
- Embeddings are randomly initialised (the paper uses skip-gram vectors trained on an OpenStack corpus; `--emb` supports them).
- Only OpenStack and Hadoop are used; no downstream anomaly detection experiments.
