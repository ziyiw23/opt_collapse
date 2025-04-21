import lmdb
import gzip
import pickle
import numpy as np
import scipy.stats
from tqdm import tqdm
import os

def get_embedding(txn, key):
    value = txn.get(key.encode())
    if not value:
        return None
    data = pickle.loads(gzip.decompress(value))
    return data['embeddings'].astype(np.float32).mean(axis=0)

def get_common_keys(path1, path2):
    def numeric_keys(env):
        with env.begin() as txn:
            return {k.decode() for k in txn.cursor().iternext(values=False) if k.decode().isdigit()}
    env1 = lmdb.open(path1, readonly=True, lock=False)
    env2 = lmdb.open(path2, readonly=True, lock=False)
    keys = sorted(numeric_keys(env1) & numeric_keys(env2))
    env1.close(); env2.close()
    return keys

def compute_centroid(env1, env2, keys):
    sum_vec = None
    count = 0
    with env1.begin() as tx1, env2.begin() as tx2:
        for key in keys:
            emb1 = get_embedding(tx1, key)
            emb2 = get_embedding(tx2, key)
            if emb1 is None or emb2 is None: continue
            vec = emb1 + emb2
            if sum_vec is None:
                sum_vec = np.zeros_like(vec)
            sum_vec += vec
            count += 2
    return sum_vec / count if count else None

def compute_metrics(orig, opt, centroid):
    o_c = orig - centroid
    p_c = opt - centroid
    o_n = o_c / (np.linalg.norm(o_c) + 1e-8)
    p_n = p_c / (np.linalg.norm(p_c) + 1e-8)
    return {
        'cosine': np.dot(o_n, p_n),
        'flipped_cosine': np.dot(o_n, -p_n),
        'pcc': scipy.stats.pearsonr(orig, opt)[0],
        'l2_diff': np.linalg.norm(orig - opt),
        'mse': np.mean((orig - opt)**2),
        'spearman': scipy.stats.spearmanr(orig, opt)[0]
    }

def compute_summary_metrics(orig_path, opt_path):
    env1 = lmdb.open(orig_path, readonly=True, lock=False)
    env2 = lmdb.open(opt_path, readonly=True, lock=False)
    keys = get_common_keys(orig_path, opt_path)
    centroid = compute_centroid(env1, env2, keys)
    all_metrics = {k: [] for k in ['cosine', 'flipped_cosine', 'pcc', 'l2_diff', 'mse', 'spearman']}
    with env1.begin() as tx1, env2.begin() as tx2:
        for key in tqdm(keys, desc=f"Comparing {os.path.basename(opt_path)}"):
            emb1 = get_embedding(tx1, key)
            emb2 = get_embedding(tx2, key)
            if emb1 is None or emb2 is None or emb1.shape != emb2.shape:
                continue
            m = compute_metrics(emb1, emb2, centroid)
            for k, v in m.items():
                all_metrics[k].append(v)
    env1.close(); env2.close()
    return {k: np.mean(v) for k, v in all_metrics.items()}
