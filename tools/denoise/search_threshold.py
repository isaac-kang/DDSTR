#!/usr/bin/env python3
"""B utility: Grid search for optimal conf_high/conf_low thresholds.

Requires a ground truth PL labels file for evaluation.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_error_info(tsv_path):
    entries = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entries.append({
                'dataset': row['dataset'],
                'sample_idx': int(row['sample_idx']),
                'pred': row['pred'],
                'gt': row['gt'],
                'r_str': float(row['pred_mean_lp']) - float(row['gt_mean_lp']),
            })
    return entries


def load_judge_results(tsv_path):
    results = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['dataset'], int(row['sample_idx']))
            results[key] = row['answer']
    return results


def load_pl_labels(tsv_path):
    labels = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            labels[(row['dataset'], int(row['sample_idx']))] = row['pl']
    return labels


def evaluate(entries, judge, pl_labels, conf_high, conf_low):
    case12_correct, case12_total = 0, 0
    total_correct = 0
    for entry in entries:
        key = (entry['dataset'], entry['sample_idx'])
        conf = sigmoid(entry['r_str'])
        if conf >= conf_high:
            generated_pl = entry['pred']
        elif conf < conf_low:
            generated_pl = entry['gt']
        else:
            answer = judge.get(key, '2')
            generated_pl = entry['pred'] if answer == '1' else entry['gt']

        gt_pl = pl_labels.get(key, '')
        if gt_pl in (entry['pred'], entry['gt']):
            case12_total += 1
            if generated_pl == gt_pl:
                case12_correct += 1
        if generated_pl == gt_pl:
            total_correct += 1

    return case12_correct, case12_total, total_correct, len(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_info', required=True, help='error_info.tsv')
    parser.add_argument('--judge', required=True, help='LLM judge results TSV')
    parser.add_argument('--pl_labels', required=True, help='Ground truth PL labels TSV')
    parser.add_argument('--metric', choices=['case12', 'total'], default='case12')
    parser.add_argument('--step', type=float, default=0.01)
    args = parser.parse_args()

    entries = load_error_info(args.error_info)
    judge = load_judge_results(args.judge)
    pl_labels = load_pl_labels(args.pl_labels)
    print(f"Loaded {len(entries)} entries, {len(judge)} judge, {len(pl_labels)} PL labels")

    lows = np.arange(0.50, 1.00, args.step)
    highs = np.arange(0.50, 1.01, args.step)

    all_results = []
    for conf_low in lows:
        for conf_high in highs:
            if conf_high <= conf_low:
                continue
            c12_c, c12_t, tot_c, tot = evaluate(entries, judge, pl_labels, conf_high, conf_low)
            if args.metric == 'case12':
                acc = c12_c / c12_t * 100 if c12_t > 0 else 0
            else:
                acc = tot_c / tot * 100 if tot > 0 else 0
            all_results.append((acc, conf_low, conf_high, c12_c, c12_t, tot_c, tot))

    all_results.sort(key=lambda x: -x[0])
    best = all_results[0]
    print(f"\nBest: conf_low={best[1]:.2f}, conf_high={best[2]:.2f}")
    print(f"  case1+2: {best[3]}/{best[4]} ({best[3]/best[4]*100:.2f}%)")
    print(f"  total:   {best[5]}/{best[6]} ({best[5]/best[6]*100:.2f}%)")

    print(f"\nTop-10:")
    print(f"  {'#':<4} {'conf_low':<10} {'conf_high':<10} {'case1+2':>12} {'total':>12}")
    for i, (acc, cl, ch, c12_c, c12_t, tot_c, tot) in enumerate(all_results[:10]):
        print(f"  {i+1:<4} {cl:<10.2f} {ch:<10.2f} {c12_c}/{c12_t} ({c12_c/c12_t*100:.2f}%)  {tot_c}/{tot} ({tot_c/tot*100:.2f}%)")

    print(f"\nUsage: python generate_pl.py --conf_high {best[2]:.2f} --conf_low {best[1]:.2f} ...")


if __name__ == '__main__':
    main()
