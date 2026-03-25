#!/usr/bin/env python3
"""B Step 3: Generate denoised PL labels and write PL LMDB.

Reads error_info.tsv and judge results (from llm_judge.py).
Uses the fused decision (r_final = α·r_STR + (1-α)·r_LLM) to select
between pred and gt for each error sample.

Then creates a new LMDB where error samples have their labels replaced
with the PL decision, while all other samples keep their original GT.

dataset keys in TSV match the relative LMDB path from data_root/train/{train_dir}/.
sample_idx is the 0-based raw LMDB index.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import lmdb as lmdb_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_error_info(tsv_path):
    """Load error_info.tsv -> dict keyed by (dataset, sample_idx)."""
    errors = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['dataset'], int(row['sample_idx']))
            errors[key] = {
                'pred': row['pred'],
                'gt': row['gt'],
                'pred_mean_lp': float(row['pred_mean_lp']),
                'gt_mean_lp': float(row['gt_mean_lp']),
            }
    return errors


def load_judge_results(tsv_path):
    """Load judge results TSV -> dict keyed by (dataset, sample_idx)."""
    results = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['dataset'], int(row['sample_idx']))
            results[key] = row['answer']
    return results


def copy_lmdb_with_pl(src_dir, dst_dir, pl_decisions, ds_name):
    """Copy an LMDB, replacing labels for samples that have a PL decision."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_env = lmdb_lib.open(str(src_dir), readonly=True, lock=False)
    src_mdb = src_dir / 'data.mdb'
    map_size = max(int(src_mdb.stat().st_size * 1.5), 1024 * 1024 * 100)
    dst_env = lmdb_lib.open(str(dst_dir), map_size=map_size)

    with src_env.begin() as src_txn:
        num_samples_raw = src_txn.get('num-samples'.encode())
        if num_samples_raw is None:
            src_env.close()
            dst_env.close()
            return 0, 0

        num_samples = int(num_samples_raw.decode())
        n_replaced = 0

        with dst_env.begin(write=True) as dst_txn:
            dst_txn.put('num-samples'.encode(), str(num_samples).encode())
            for idx in range(1, num_samples + 1):
                img_key = f'image-{idx:09d}'.encode()
                label_key = f'label-{idx:09d}'.encode()
                img_data = src_txn.get(img_key)
                label_data = src_txn.get(label_key)
                if img_data is None or label_data is None:
                    continue
                dst_txn.put(img_key, img_data)
                # sample_idx is 0-based, LMDB idx is 1-based
                key = (ds_name, idx - 1)
                if key in pl_decisions:
                    dst_txn.put(label_key, pl_decisions[key].encode())
                    n_replaced += 1
                else:
                    dst_txn.put(label_key, label_data)

    src_env.close()
    dst_env.close()
    return num_samples, n_replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--error_info', required=True, help='error_info.tsv')
    parser.add_argument('--judge', required=True, help='LLM judge results TSV')
    parser.add_argument('--data_root', required=True, help='Original LMDB root dir')
    parser.add_argument('--output_root', required=True, help='Output PL LMDB root dir')
    parser.add_argument('--train_dirs', nargs='+',
                        default=['Union14M-L-LMDB-Filtered/filter_train_challenging',
                                 'Union14M-L-LMDB-Filtered/filter_train_hard',
                                 'Union14M-L-LMDB-Filtered/filter_train_medium',
                                 'Union14M-L-LMDB-Filtered/filter_train_normal',
                                 'Union14M-L-LMDB-Filtered/filter_train_easy'],
                        help='Train directories relative to data_root')
    args = parser.parse_args()

    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.output_root = str(Path(args.output_root).expanduser().resolve())

    errors = load_error_info(args.error_info)
    judge = load_judge_results(args.judge)
    print(f"Loaded {len(errors)} errors, {len(judge)} judge results")

    # Apply PL decisions
    pl_decisions = {}
    stats = {'pred': 0, 'gt': 0}
    for key, err in errors.items():
        answer = judge.get(key, '2')  # default to gt if no judge result
        if answer == '1':
            pl_decisions[key] = err['pred']
            stats['pred'] += 1
        else:
            pl_decisions[key] = err['gt']
            stats['gt'] += 1

    print(f"\nPL decisions (based on r_final):")
    print(f"  -> pred: {stats['pred']}")
    print(f"  -> gt:   {stats['gt']}")

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    total_samples_all = 0
    total_replaced_all = 0
    per_dataset_stats = []

    for train_dir in args.train_dirs:
        src_train_root = data_root / train_dir
        dst_train_root = output_root / train_dir

        lmdb_dirs = sorted(p.parent for p in src_train_root.rglob('data.mdb'))
        if not lmdb_dirs:
            # Fallback: try data_root/train/{train_dir} for backward compat
            src_train_root2 = data_root / 'train' / train_dir
            lmdb_dirs = sorted(p.parent for p in src_train_root2.rglob('data.mdb'))
            if lmdb_dirs:
                src_train_root = src_train_root2
                dst_train_root = output_root / 'train' / train_dir
            else:
                print(f"No LMDBs found under {src_train_root}")
                continue

        for lmdb_dir in lmdb_dirs:
            ds_name = str(lmdb_dir.relative_to(data_root))
            dst_dir = output_root / lmdb_dir.relative_to(data_root)

            n_total, n_replaced = copy_lmdb_with_pl(lmdb_dir, dst_dir, pl_decisions, ds_name)
            total_samples_all += n_total
            total_replaced_all += n_replaced
            per_dataset_stats.append({
                'dataset': ds_name,
                'total_samples': n_total,
                'replaced': n_replaced,
                'replace_ratio': round(n_replaced / n_total, 6) if n_total > 0 else 0.0,
            })
            print(f"  {ds_name}: {n_total} samples, {n_replaced} replaced "
                  f"({n_replaced/n_total*100:.2f}%)" if n_total > 0 else f"  {ds_name}: empty")

    # Save stats
    overall_ratio = total_replaced_all / total_samples_all if total_samples_all > 0 else 0.0
    stats_summary = {
        'total_samples': total_samples_all,
        'total_replaced': total_replaced_all,
        'overall_replace_ratio': round(overall_ratio, 6),
        'per_dataset': per_dataset_stats,
    }
    stats_path = output_root / 'pl_stats.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\nPL LMDB written to {output_root}")
    print(f"Label change ratio: {total_replaced_all}/{total_samples_all} "
          f"({overall_ratio*100:.2f}%) sequences changed")
    print(f"Stats saved to {stats_path}")


if __name__ == '__main__':
    main()
