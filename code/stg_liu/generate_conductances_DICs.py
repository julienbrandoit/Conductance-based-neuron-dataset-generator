import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from stg import generate_neuromodulated_population
import time
import argparse


def generate_chunk(args):
    i, M = args

    g_s_range = [-20, 20]
    g_u_range = [0, 20]
    V_th = -51.0

    seed = (int(time.time_ns()) + os.getpid() + i) % 2**32
    np.random.seed(seed)

    g_s = np.random.uniform(g_s_range[0], g_s_range[1])
    g_u = np.random.uniform(g_u_range[0], g_u_range[1])

    pop = generate_neuromodulated_population(M, V_th, g_s, g_u, iterations=5)

    actual_size = pop.shape[0]

    data = {
        'ID': np.arange(i * M, i * M + actual_size),
        'g_s': [g_s] * actual_size,
        'g_u': [g_u] * actual_size,
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
    parser = argparse.ArgumentParser(description="Generate STG dataset conductances via the DICs framework.")
    parser.add_argument("--P", type=int, default=62500, help="Number of chunks (each with M neurons sharing g_s, g_u)")
    parser.add_argument("--M", type=int, default=16, help="Population size per chunk")
    parser.add_argument("--max_workers", type=int, default=19, help="Maximum number of parallel workers")
    parser.add_argument("--output_file", type=str, default="stg_dataset_conductances.csv", help="Output CSV file name")
    args = parser.parse_args()

    P = args.P
    M = args.M
    MAX_WORKERS = args.max_workers
    output_file = args.output_file

    print(f"Generating {P*M} samples ({P} populations of size {M}) with {MAX_WORKERS} workers.")

    all_dfs = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_chunk, (i, M)) for i in range(P)]

        with tqdm(total=P, file=sys.stdout, disable=not sys.stdout.isatty()) as pbar:
            for future in as_completed(futures):
                all_dfs.append(future.result())
                pbar.update(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
