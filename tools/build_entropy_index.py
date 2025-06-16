#!/usr/bin/env python3
"""
Compute local byte entropy over a sliding window and produce an index file.
Outputs two arrays:
  offsets: list of window start positions
  entropies: corresponding bits-per-byte in each window

Usage:
  python tools/build_entropy_index.py \
      --input data/tiny.bin \
      --window 256 \
      --stride 64 \
      --output data/tiny_entropy.npz

"""
import argparse
import pathlib
import numpy as np
import math

def compute_entropy(window_bytes: np.ndarray) -> float:
    # window_bytes: 1D array of uint8
    counts = np.bincount(window_bytes, minlength=256)
    probs = counts / counts.sum()
    # avoid log(0): mask non-zero
    nz = probs > 0
    return -np.sum(probs[nz] * np.log2(probs[nz]))


def build_index(data: bytes, window: int, stride: int):
    # Convert to numpy array of uint8
    arr = np.frombuffer(data, dtype=np.uint8)
    length = arr.size
    offsets = []
    entropies = []
    for start in range(0, length - window + 1, stride):
        win = arr[start:start+window]
        h = compute_entropy(win)
        offsets.append(start)
        entropies.append(h)
    return np.array(offsets, dtype=np.int64), np.array(entropies, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Build byte-entropy index for a binary dataset.")
    p.add_argument("--input",  type=str, required=True, help="Path to input .bin file.")
    p.add_argument("--window", type=int, default=256, help="Window size in bytes.")
    p.add_argument("--stride", type=int, default=64, help="Stride between windows.")
    p.add_argument("--output", type=str, required=True, help="Output .npz file path.")
    args = p.parse_args()

    data_path = pathlib.Path(args.input)
    out_path  = pathlib.Path(args.output)

    print(f"🔍 Reading input file: {data_path} ({data_path.stat().st_size} bytes)")
    data = data_path.read_bytes()

    print(f"⚙  Computing entropy with window={args.window}, stride={args.stride}...")
    offsets, entropies = build_index(data, args.window, args.stride)

    print(f"💾 Saving index to: {out_path}")
    np.savez_compressed(out_path, offsets=offsets, entropies=entropies)
    print(f"✅ Done: {offsets.size} windows computed.")

if __name__ == '__main__':
    main()
