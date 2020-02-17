#!/usr/bin/env python3
import sys
from collections import defaultdict
try:
    import ujson as json
except:
    import json


CHMAP = {c: i for i, c in enumerate('NATCG')}
ENCDICT =  defaultdict(int)
def ENC(s):
    if s not in ENCDICT:
        for c in s:
            ENCDICT[s] = ENCDICT[s] * 5 + CHMAP[c]
    return ENCDICT[s]

if __name__ == '__main__':
    if len(sys.argv) < 1 + 5:
        print('--usage {} pos_data neg_data K L output_file'.format(
            sys.argv[0]), file=sys.stderr)
        sys.exit(0)
    pos_file, neg_file, K, L, output_file = sys.argv[1:]
    L = int(L)

    LL, LR = (L - 1) // 2, (L + 2) // 2
    assert(LL + LR == L)
    K = int(K)
    KL, KR = (K - 1) // 2, (K + 2) // 2
    assert(KL + KR == K)

    PL = K + L * 2
    PAD = 'N' * PL
    
    tmpl = 'pos_data = {}\nneg_data = {}\nL = {}\nK = {}\noutput_file = {}'

    print(tmpl.format(pos_file, neg_file, L, K, output_file), file=sys.stderr)

    with open(output_file, 'w') as wp:
        for file_name in [pos_file, neg_file]:
            print('Processing {}.'.format(file_name), file=sys.stderr)
            lbl = 1 if file_name == pos_file else 0
            with open(file_name, 'r') as fp:
                for line in fp:
                    data = json.loads(line)
                    # Pre-process.
                    seg = PAD + data['seq'] + PAD
                    ks = [ENC(seg[(i - KL):(i + KR)]) for i in range(len(seg))]
                    if data['strand'] == '-': ks.reverse()
                    # Extract.
                    acceptors = [ks[(i + PL - LL):(i + PL + LR)] \
                            for i in data['junctions']['head']]
                    donors = [ks[(i + PL - LL):(i + PL + LR)] \
                            for i in data['junctions']['tail']]
                    # Validation.
                    for x in acceptors:
                        assert(len(x) == L)
                    for x in donors:
                        assert(len(x) == L)

                    if data['strand'] == '-':
                        for x in acceptors: x.reverse()
                        for x in donors: x.reverse()

                    print(json.dumps({
                        'acceptors': acceptors,
                        'donors': donors,
                        'label': lbl}), file=wp) 
