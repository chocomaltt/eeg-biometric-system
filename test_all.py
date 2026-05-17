#!/usr/bin/env python3
"""Batch testing script for EEG biometric models with Google Sheets logging."""

import os
import re
import json
import subprocess
from itertools import product
from sheets_connector import log_to_sheets

# Configurations to test
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
SEEDS = [123, 42]

NOTEBOOK_PATH = 'notebooks/03_ann_testing.ipynb'


def parse_metrics_from_notebook():
    """Parse metrics from the executed notebook's output cells."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Find the output containing the evaluation results
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        outputs = cell.get('outputs', [])
        for output in outputs:
            text = ''
            if output.get('output_type') == 'stream':
                text = ''.join(output.get('text', []))
            elif output.get('output_type') == 'execute_result':
                text = ''.join(output.get('data', {}).get('text/plain', []))

            if 'EUCLIDEAN DISTANCE EVALUATION RESULTS' in text:
                metrics = {}

                # Parse Top-1 Accuracy
                match = re.search(r'Top-1 Accuracy\s*:\s*([\d.]+)%', text)
                if match:
                    metrics['top1_accuracy'] = float(match.group(1)) / 100

                # Parse EER
                match = re.search(r'EER\s*:\s*([\d.]+)%', text)
                if match:
                    metrics['eer'] = float(match.group(1)) / 100

                # Parse EER Threshold
                match = re.search(r'EER Threshold \(distance\)\s*:\s*([\d.]+)', text)
                if match:
                    metrics['eer_threshold'] = float(match.group(1))

                if len(metrics) == 3:
                    return metrics

    return None


def run_testing(data_type, window_size, stride, seed):
    """Run testing notebook with specific config."""
    env = os.environ.copy()
    env['DATA_TYPE'] = data_type
    env['WINDOW_SIZE'] = str(window_size)
    env['STRIDE'] = str(stride)
    env['SEEDER'] = str(seed)

    print(f"\n{'='*60}")
    print(f"Testing: {data_type.upper()} | Window: {window_size}s | Stride: {stride}s | Seed: {seed}")
    print(f"{'='*60}\n")

    cmd = [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', NOTEBOOK_PATH,
        '--ExecutePreprocessor.timeout=3600'
    ]

    result = subprocess.run(
        cmd,
        env=env,
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        print(f"WARNING: Testing failed for {data_type} w={window_size} s={stride} seed={seed}")
        return None

    # Parse metrics from the executed notebook
    metrics = parse_metrics_from_notebook()

    if not metrics:
        print(f"WARNING: Could not parse metrics for {data_type} w={window_size} s={stride} seed={seed}")
    else:
        print(f"Metrics: Acc={metrics['top1_accuracy']*100:.2f}% | EER={metrics['eer']*100:.2f}% | Threshold={metrics['eer_threshold']:.4f}")

    return metrics


def main():
    results = []

    for seed, (window_size, stride), data_type in product(SEEDS, WINDOW_CONFIGS, DATA_TYPES):
        metrics = run_testing(data_type, window_size, stride, seed)

        if metrics:
            # Log to Google Sheets
            try:
                log_to_sheets(
                    data_type=data_type,
                    window_size=window_size,
                    stride=stride,
                    top1_accuracy=metrics['top1_accuracy'],
                    eer_threshold=metrics['eer_threshold'],
                    eer=metrics['eer'],
                    seed=seed
                )
                print(f"Logged to Google Sheets: {data_type} w={window_size} s={stride} seed={seed}")
            except Exception as e:
                print(f"Failed to log to sheets: {e}")

        results.append({
            'data_type': data_type,
            'window': window_size,
            'stride': stride,
            'seed': seed,
            'success': metrics is not None,
            'metrics': metrics
        })

    print("\n" + "="*60)
    print("TESTING COMPLETE - SUMMARY")
    print("="*60)
    for r in results:
        status = "OK" if r['success'] else "FAILED"
        metrics_str = ""
        if r['metrics']:
            metrics_str = f" | Acc: {r['metrics']['top1_accuracy']*100:.2f}% | EER: {r['metrics']['eer']*100:.2f}%"
        print(f"  {r['data_type'].upper()} w={r['window']} s={r['stride']} seed={r['seed']}: {status}{metrics_str}")


if __name__ == '__main__':
    main()
