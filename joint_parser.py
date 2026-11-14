"""Joint parser: domain knowledge database + implicit CI pair discovery + conceptualized template."""
from data import CONCEPT, INSTANCE


class DomainKnowledge:
    def __init__(self):
        self.db = {}          # instance -> concept (only high-quality pairs)

    def add(self, concept, instance):
        self.db[instance] = concept

    def find(self, instance):
        return self.db.get(instance)


def joint_parse(tokens, types, pairs, dk):
    """tokens/types: per-token lists (no <TMP>). pairs: explicit (concept_idx, instance_idx) from the miner."""
    ci, used_c, used_i = [], set(), set()
    for j, i in pairs:                                   # step 1: store explicit pairs
        if types[j] == CONCEPT and types[i] == INSTANCE:
            dk.add(tokens[j], tokens[i])
            ci.append((tokens[j], tokens[i], i)); used_c.add(j); used_i.add(i)
    for i, t in enumerate(types):                        # step 2: implicit pairs from history
        if t == INSTANCE and i not in used_i:
            c = dk.find(tokens[i])
            if c:
                ci.append((c, tokens[i], i)); used_i.add(i)
    cmap = {i: c for c, _, i in ci}
    template = ['<%s>' % cmap[k] if k in cmap else '<*>' if types[k] == INSTANCE else tokens[k]
                for k in range(len(tokens))]
    return {'template': ' '.join(template),
            'ci_pairs': [(c, v) for c, v, _ in ci],
            'orphan_concepts': [tokens[j] for j, t in enumerate(types) if t == CONCEPT and j not in used_c],
            'orphan_instances': [tokens[i] for i, t in enumerate(types) if t == INSTANCE and i not in used_i]}


if __name__ == '__main__':
    dk = DomainKnowledge()
    dk.add('cell', '949e1227')
    toks = 'Lock 949e1227 acquired by nova.context()'.split()
    print(joint_parse(toks, [0, 2, 0, 0, 2], [], dk))
