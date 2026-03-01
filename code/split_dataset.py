import os
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a merged dataset CSV into train and val sets."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to merged dataset CSV",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation set (e.g., 0.1 = 10%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"{args.input_file} does not exist")

    if not (0.0 < args.frac < 1.0):
        raise ValueError("--frac must be between 0 and 1")

    print(f"Loading dataset from {args.input_file}...")
    df = pd.read_csv(args.input_file)

    print(f"Total rows: {len(df)}")

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Split
    val_size = int(len(df) * args.frac)
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]

    # Generate output filenames
    base, ext = os.path.splitext(args.input_file)
    train_file = f"{base}_train{ext}"
    val_file = f"{base}_val{ext}"

    # Save
    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)

    print("Split complete:")
    print(f"Train set: {len(df_train)} rows -> {train_file}")
    print(f"Val set:   {len(df_val)} rows -> {val_file}")