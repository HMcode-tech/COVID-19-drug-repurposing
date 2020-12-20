import sys
sys.path.insert(1, '../utils')
from utils import download_and_extract
download_and_extract()
drkg_file = 'drkg.tsv'

import pandas as pd
import numpy as np

df = pd.read_csv(drkg_file, sep="\t")
triples = df.values.tolist()
num_triples=len(triples)
seed = np.arange(num_triples)
np.random.shuffle(seed)
train_cnt = int(num_triples * 0.9)
valid_cnt = int(num_triples * 0.05)
train_set = seed[:train_cnt]
train_set = train_set.tolist()
valid_set = seed[train_cnt:train_cnt+valid_cnt].tolist()
test_set = seed[train_cnt+valid_cnt:].tolist()
with open("train/drkg_train.tsv", 'w+') as f:
    for idx in train_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))
                        
with open("train/drkg_valid.tsv", 'w+') as f:
    for idx in valid_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

with open("train/drkg_test.tsv", 'w+') as f:
    for idx in test_set:
        f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))
