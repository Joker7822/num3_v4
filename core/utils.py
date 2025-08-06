import numpy as np
import pandas as pd
import random

def preprocess_data(df):
    print("[INFO] Preprocessing data...")
    return np.random.rand(len(df), 10), np.random.randint(0, 9, size=(len(df), 3)), None

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def parse_number_string(s):
    return [int(c) for c in str(s) if c.isdigit()]
