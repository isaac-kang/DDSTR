"""Save PL=pred images grouped by LLM-only softmax(score) bins.

For each score bin (0.0~0.2, 0.2~0.4, ..., 0.8~1.0), saves images with
pred and gt in the filename so you can visually inspect them.

Usage:
    python tools/analysis/save_score_bin_images.py
    python tools/analysis/save_score_bin_images.py --eval_root ~/data/STR/openocr/evaluation --output_dir output/analysis/score_bins
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sanitize(s, max_len=30):
    """Sanitize string for use in filename."""
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)[:max_len]


def load_image_from_lmdb(eval_root, dataset_name, image_index):
    lmdb_dir = os.path.join(eval_root, dataset_name)
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        img_key = f'image-{image_index:09d}'.encode()
        imgbuf = txn.get(img_key)
        imgbuf = bytes(imgbuf)
    env.close()
    return imgbuf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='tools/analysis/PL_oracle.csv')
    parser.add_argument('--llm_cache', default='tools/analysis/llm_scores_cache_Qwen-Qwen3-32B-AWQ.json')
    parser.add_argument('--eval_root', default='~/data/STR/openocr/evaluation')
    parser.add_argument('--output_dir', default='output/analysis/score_bins_llm')
    parser.add_argument('--no_name', action='store_true',
                        help='Omit pred/gt from filenames (saves to score_bins_llm_no_name)')
    parser.add_argument('--max_per_bin', type=int, default=0,
                        help='Max images per bin (0 = save all)')
    args = parser.parse_args()
    if args.no_name and args.output_dir == 'output/analysis/score_bins_llm':
        args.output_dir = 'output/analysis/score_bins_llm_no_name'
    args.eval_root = str(Path(args.eval_root).expanduser().resolve())

    # Load CSV
    entries = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pl = row['PL'] if row['PL'] else ''
            pred = row['pred']
            if pl == pred:  # PL=pred only
                entries.append({
                    'dataset_name': row['dataset_name'],
                    'image_index': int(row['image_index']),
                    'pred': pred,
                    'gt': row['gt'],
                })
    print(f'PL=pred entries: {len(entries)}')

    # Load LLM scores (support both old list and new {llm_model, scores} format)
    with open(args.llm_cache, 'r') as f:
        cache = json.load(f)
    scores_list = cache if isinstance(cache, list) else cache['scores']
    cache_lookup = {(r['dataset_name'], r['image_index']): r['r_llm'] for r in scores_list}

    # Compute conf_llm = sigmoid(r_llm) for each entry
    for e in entries:
        r_llm = cache_lookup.get((e['dataset_name'], e['image_index']), 0.0)
        e['conf_llm'] = float(sigmoid(r_llm))

    # Define bins
    bin_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bins = {f'{lo:.1f}-{hi:.1f}': [] for lo, hi in bin_edges}

    for e in entries:
        conf = e['conf_llm']
        for lo, hi in bin_edges:
            if lo <= conf < hi or (hi == 1.0 and conf == 1.0):
                bins[f'{lo:.1f}-{hi:.1f}'].append(e)
                break

    # Print stats
    print(f'\n{"bin":<12} {"count":>6}')
    print('-' * 20)
    for name, items in bins.items():
        print(f'{name:<12} {len(items):>6}')

    # Save images
    for bin_name, items in bins.items():
        bin_dir = os.path.join(args.output_dir, bin_name)
        os.makedirs(bin_dir, exist_ok=True)

        save_items = items
        if args.max_per_bin > 0:
            save_items = items[:args.max_per_bin]

        for i, e in enumerate(save_items):
            # image_index is 0-based in CSV, LMDB keys are 1-based
            imgbuf = load_image_from_lmdb(args.eval_root, e['dataset_name'], e['image_index'] + 1)
            img = Image.open(io.BytesIO(imgbuf))

            conf_str = f"{e['conf_llm']:.3f}"
            if args.no_name:
                fname = f"{i:03d}_conf{conf_str}.png"
            else:
                pred_s = sanitize(e['pred'])
                gt_s = sanitize(e['gt'])
                fname = f"{i:03d}_conf{conf_str}_pred_{pred_s}_gt_{gt_s}.png"
            img.save(os.path.join(bin_dir, fname))

        print(f'Saved {len(save_items)} images to {bin_dir}')

    print(f'\nDone. Output: {args.output_dir}')


if __name__ == '__main__':
    main()
