import json
import random
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(SEED, DETERMINISTIC=False):
    random.seed(SEED)  # python random seed 고정
    np.random.seed(SEED)  # numpy random seed 고정
    torch.manual_seed(SEED)  # torch random seed 고정
    torch.cuda.manual_seed_all(SEED)
    if DETERMINISTIC:  # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_rows(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True) ## numpy array를 행별로 표준화


def zero_one_normalize_rows(X): ## numpy array에 대한 행별 0-1 표준화
    min_by_rows = X.min(axis=1, keepdims=True)
    max_by_rows = X.max(axis=1, keepdims=True)
    return (X - min_by_rows) / (max_by_rows - min_by_rows)
