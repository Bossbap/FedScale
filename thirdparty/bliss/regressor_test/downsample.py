#!/usr/bin/env python3
"""
Down-sample the 'g' dataset by randomly deleting half the points
in rounds 100–250 (inclusive).

Usage:
  python downsample.py \
    --data_dir thirdparty/bliss/regressor_test/datasets/femnist \
    [--start 100] [--end 250] [--seed 42]
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    p = argparse.ArgumentParser(
        description="Down-sample g_*.csv by halving data in specific rounds."
    )
    p.add_argument(
        "--data_dir", required=True,
        help="…/regressor_test/datasets/<job_name> (folder with g_*.csv)"
    )
    p.add_argument(
        "--start", type=int, default=100,
        help="First round to down-sample (inclusive)"
    )
    p.add_argument(
        "--end", type=int, default=250,
        help="Last round to down-sample (inclusive)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    # pick the latest g_*.csv
    csvs = sorted(data_dir.glob("g_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No g_*.csv found in {data_dir}")
    src = csvs[-1]
    print(f"Loading {src.name}…")
    df = pd.read_csv(src)

    # mask of rows in the target round interval
    m = df["round"].between(args.start, args.end)
    to_down = df[m]
    to_keep = df[~m]

    rng = np.random.RandomState(args.seed)
    # sample half of the indices in 'to_down'
    keep_idx = rng.choice(
        to_down.index,
        size=len(to_down) // 2,
        replace=False
    )

    # reconstruct the down-sampled DataFrame
    df_ds = pd.concat([to_keep, df.loc[keep_idx]], axis=0).sort_index()

    # overwrite original file in-place
    out = src
    print(f"Overwriting original file: {out.name}…")
    df_ds.to_csv(out, index=False)
    print("Done.")

if __name__ == "__main__":
    main()