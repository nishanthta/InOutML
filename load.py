from __future__ import print_function

import codecs
import csv
import gzip
from nltk.tokenize import word_tokenize
import numpy as np
import json

def load_sts(dsfile, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        cnt = 0
        for line in f:
            cnt+=1
            if cnt == 1:
                continue
            line = line.rstrip()
            _, s0x, s1x, label,_ = line.split('\t')
            if label == '':
                if skip_unlabeled:
                    continue
                else:
                    labels.append(-1.)
            else:
                labels.append(float(label))
            s0.append(s0x)
            s1.append(s1x)
            # cnt+=1
    return (s0, s1, labels)