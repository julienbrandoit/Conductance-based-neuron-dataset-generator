import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse


def sample_individual(M, distribution='uniform'):
    g_Na_range = [0, 8000]
    g_Kd_range = [0, 350]
    g_CaT_range = [0, 12]
    g_CaS_range = [0, 50]
    g_KCa_range = [0, 250]
    g_A_range = [0, 600]
    g_H_range = [0, 0.7]
    g_leak_range = [0, 0.02]

    ranges = [g_Na_range, g_Kd_range, g_CaT_range, g_CaS_range,
              g_KCa_range, g_A_range, g_H_range, g_leak_range]

    if distribution == 'uniform':
        samples = np.column_stack([
            np.random.uniform(r[0], r[1], M) for r in ranges
        ])
    elif distribution == 'gamma':
        means = [(r[0] + r[1]) / 2 for r in ranges]
        stds = [np.sqrt((r[1] - r[0])**2 / 12) for r in ranges]
        shapes = [(mean / std)**2 for mean, std in zip(means, stds)]
        scales = [std**2 / mean for mean, std in zip(means, stds)]
        samples = np.column_stack([
            np.random.gamma(shapes[i], scales[i], M) for i in range(len(ranges))
        ])
    else:
        raise ValueError("Unsupported distribution. Use 'uniform' or 'gamma'.")

    return samples


def generate_chunk(args):
    i, M, offset, distribution = args

    seed = (int(time.time_ns()) + os.getpid() + i) % 2**32
    np.random.seed(seed)

    pop = sample_individual(M, distribution=distribution)

    data = {
        'ID': np.arange(offset, offset + M),
        'g_Na': pop[:, 0],
        'g_Kd': pop[:, 1],
        'g_CaT': pop[:, 2],
        'g_CaS': pop[:, 3],
        'g_KCa': pop[:, 4],
        'g_A': pop[:, 5],
        'g_H': pop[:, 6],
        'g_leak': pop[:, 7],
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate STG dataset conductances via Monte Carlo sampling.")
    parser.add_argument("--P", type=int, default=1000000, help="Total number of samples to generate")
    parser.add_argument("--M", type=int, default=1, help="Internal batch size (samples per parallel task)")
    parser.add_argument("--max_workers", type=int, default=19, help="Maximum number of parallel workers")
    parser.add_argument("--output_file", type=str, default="stg_dataset_conductances.csv", help="Output CSV file name")
    parser.add_argument("--distribution", type=str, default="uniform", choices=["uniform", "gamma"],
                        help="Sampling distribution for conductances")
    args = parser.parse_args()

    P = args.P
    M = args.M
    MAX_WORKERS = args.max_workers
    output_file = args.output_file

    n_chunks = math.ceil(P / M)
    # build chunk sizes so the last chunk is trimmed if P % M != 0
    chunk_sizes = [M] * n_chunks
    if P % M != 0:
        chunk_sizes[-1] = P % M

    offsets = [sum(chunk_sizes[:i]) for i in range(n_chunks)]

    print(f"Generating {P} samples with {MAX_WORKERS} workers ({args.distribution} distribution).")

    all_dfs = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(generate_chunk, (i, chunk_sizes[i], offsets[i], args.distribution))
            for i in range(n_chunks)
        ]

        with tqdm(total=n_chunks, file=sys.stdout, disable=not sys.stdout.isatty()) as pbar:
            for future in as_completed(futures):
                all_dfs.append(future.result())
                pbar.update(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values('ID').reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")
