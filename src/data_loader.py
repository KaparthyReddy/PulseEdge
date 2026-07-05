import os
import scipy.io as sio
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_mat_file(filename):
    path = os.path.join(DATA_DIR, filename)
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if "DE_time" in k]
    if not keys:
        raise ValueError(f"No DE_time key found in {filename}")
    return mat[keys[0]].flatten()

def load_all_signals():
    signals = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".mat"):
            label = fname.replace(".mat", "")
            signals[label] = load_mat_file(fname)
    return signals
