#!/usr/bin/env python3
"""Batch training script for EEG biometric models."""

import os
import sys
import subprocess
from itertools import product

# Configurations to train
DATA_TYPES = ['eo', 'ec']
WINDOW_CONFIGS = [
    [1, 0.5],
    [1, 1],
    [1.5, 0.5],
    [1.5, 1],
    [1.5, 1.5],
    [2, 0.5],
    [2, 1],
    [2, 1.5],
    [2, 2]
]
# SEEDS = [42, 0, 1, 123, 2024, 7, 13, 99, 1337, 314]
SEEDS = [0, 1, 123, 2024, 7, 13, 99, 1337, 314]

def run_training(data_type, window_size, stride, seed):
    """Run training notebook with specific config."""
    env = os.environ.copy()
    env['DATA_TYPE'] = data_type
    env['WINDOW_SIZE'] = str(window_size)
    env['STRIDE'] = str(stride)
    env['SEED'] = str(seed)

    print(f"\n{'='*60}")
    print(f"Training: {data_type.upper()} | Window: {window_size}s | Stride: {stride}s | Seed: {seed}")
    print(f"{'='*60}\n")

    cmd = [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', 'notebooks/02_model_training.ipynb',
        '--ExecutePreprocessor.timeout=3600'
    ]

    result = subprocess.run(cmd, env=env, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"WARNING: Training failed for {data_type} w={window_size} s={stride} seed={seed}")
        return False
    return True

def main():
    results = []

    for seed, (window_size, stride), data_type in product(SEEDS, WINDOW_CONFIGS, DATA_TYPES):
        success = run_training(data_type, window_size, stride, seed)
        results.append({
            'data_type': data_type,
            'window': window_size,
            'stride': stride,
            'seed': seed,
            'success': success
        })

    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    for r in results:
        status = "OK" if r['success'] else "FAILED"
        print(f"  {r['data_type'].upper()} w={r['window']} s={r['stride']} seed={r['seed']}: {status}")

if __name__ == '__main__':
    main()
