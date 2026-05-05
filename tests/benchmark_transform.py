import time
import numpy as np
import pandas as pd
from collapse.embedding_utils import BaseTransform


def make_df(n=128):
    rng = np.random.default_rng(0)
    elems = np.array(['C', 'N', 'O', 'S'])
    return pd.DataFrame({
        'x': rng.normal(size=n),
        'y': rng.normal(size=n),
        'z': rng.normal(size=n),
        'element': elems[rng.integers(0, len(elems), size=n)],
        'id': ['bench'] * n,
    })


def main(iters=50):
    tfm = BaseTransform(device='cpu')
    df = make_df()
    t0 = time.perf_counter()
    ok = 0
    for _ in range(iters):
        g = tfm(df)
        ok += int(g is not None)
    dt = time.perf_counter() - t0
    print(f'graphs={ok}/{iters} total_s={dt:.4f} per_graph_ms={(dt/iters)*1000:.2f}')


if __name__ == '__main__':
    main()
