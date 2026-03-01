import os
import re
import sys
import argparse
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge chunk CSVs into a single dataset.")
    parser.add_argument("--input_dir", type=str, default="chunks",
                        help="Directory containing chunk_*.csv files")
    parser.add_argument("--output_file", type=str, default="dataset.csv",
                        help="Output merged CSV filename")
    args = parser.parse_args()

    all_files = [f for f in os.listdir(args.input_dir) if f.startswith("chunk_") and f.endswith(".csv")]

    def extract_id(name):
        match = re.search(r"chunk_(\d+)\.csv", name)
        return int(match.group(1)) if match else -1

    all_files.sort(key=extract_id)

    if not all_files:
        raise FileNotFoundError(f"No chunk_*.csv files found in {args.input_dir}")

    tqdm.write(f"Found {len(all_files)} chunk files, merging...")

    dfs = []
    for f in tqdm(all_files, file=sys.stdout, disable=not sys.stdout.isatty()):
        dfs.append(pd.read_csv(os.path.join(args.input_dir, f)))

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(args.output_file, index=False)

    tqdm.write(f"Merged {len(all_files)} chunks into {args.output_file} ({len(merged)} total rows)")
