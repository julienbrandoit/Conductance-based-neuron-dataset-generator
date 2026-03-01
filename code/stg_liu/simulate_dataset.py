import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from stg import get_default_u0, get_default_parameters, simulate_individual_t_eval, ODEs, ODEs_with_noisy_current
from utils import get_spiking_times

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Global variables set by command-line arguments
SIGMA_NOISE = 0.0
ODE_FUNC = None
CUTOFF_FREQ = 1000.0


def simulate_individual(conductances):
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak = conductances
    t_simu = [0.0, 6000.0]
    t_transient = 3000.0
    dt = 0.1
    u0 = get_default_u0()
    params = get_default_parameters()
    t_eval = np.arange(t_transient, t_simu[1], dt)
    args = (u0, [g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak], t_eval, params)
    t, V = simulate_individual_t_eval(args, ode_func=ODE_FUNC, sigma_noise=SIGMA_NOISE,
                                      cutoff_freq=CUTOFF_FREQ)
    _, spiking_times = get_spiking_times(t, V)
    return f"[{', '.join(map(str, spiking_times))}]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate STG dataset chunk.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV with conductances")
    parser.add_argument("--output_dir", type=str, default="chunks", help="Directory to save chunk CSVs")
    parser.add_argument("--chunk_id", type=int, required=True, help="ID of this chunk (0-based)")
    parser.add_argument("--total_chunks", type=int, required=True, help="Total number of chunks/jobs")
    parser.add_argument("--n_workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel worker processes")
    parser.add_argument("--ode_type", type=str, default="standard", choices=["standard", "noisy"],
                        help="'standard' for clean simulation, 'noisy' for band-limited noisy current injection")
    parser.add_argument("--sigma_noise", type=float, default=0.0,
                        help="Std of Gaussian noise for injected current (only with --ode_type=noisy)")
    parser.add_argument("--cutoff_freq", type=float, default=1000.0,
                        help="Lowpass cutoff frequency in Hz for noise (only with --ode_type=noisy)")
    args = parser.parse_args()

    if args.ode_type == "noisy":
        ODE_FUNC = ODEs_with_noisy_current
        SIGMA_NOISE = args.sigma_noise
        CUTOFF_FREQ = args.cutoff_freq
        tqdm.write(f"Noisy current injection: sigma_noise={SIGMA_NOISE}, cutoff_freq={CUTOFF_FREQ} Hz")
    else:
        ODE_FUNC = ODEs
        tqdm.write("Standard ODE (no noise)")

    tqdm.write("Integration method: BDF")

    df = pd.read_csv(args.input_file)

    n = len(df)
    base_size = n // args.total_chunks
    remainder = n % args.total_chunks
    start = args.chunk_id * base_size + min(args.chunk_id, remainder)
    end = start + base_size + (1 if args.chunk_id < remainder else 0)
    df_chunk = df.iloc[start:end].copy()

    tqdm.write(f"[Chunk {args.chunk_id+1}/{args.total_chunks}] Rows {start}-{end-1} ({len(df_chunk)} samples)")

    conductances_list = df_chunk[
        ['g_Na', 'g_Kd', 'g_CaT', 'g_CaS', 'g_KCa', 'g_A', 'g_H', 'g_leak']
    ].to_numpy()

    with Pool(processes=args.n_workers) as pool:
        spiking_times = list(tqdm(
            pool.imap(simulate_individual, conductances_list),
            total=len(conductances_list),
            file=sys.stdout,
            disable=not sys.stdout.isatty()
        ))

    df_chunk['spiking_times'] = spiking_times
    os.makedirs(args.output_dir, exist_ok=True)
    chunk_path = os.path.join(args.output_dir, f"chunk_{args.chunk_id}.csv")
    df_chunk.to_csv(chunk_path, index=False)
    tqdm.write(f"[Chunk {args.chunk_id+1}] Saved to {chunk_path}")
