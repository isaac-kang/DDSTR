"""Compare two LMDB datasets: verify same images, report label diff ratio, save diff samples.

Usage:
    python tools/analysis/compare_lmdb.py <lmdb_a> <lmdb_b> [--output_dir OUTPUT_DIR]

Example:
    python tools/analysis/compare_lmdb.py \
        ~/data/STR/openocr/Union14M-L-LMDB-Filtered/filter_train_challenging \
        ~/data/STR/ddstr/SLD/mdiff4str__Qwen3-8B_a0.5/Union14M-L-LMDB-Filtered/filter_train_challenging \
        --output_dir ./output/analysis/sld_vs_original
"""

import argparse
import io
import os
import random

import lmdb
from PIL import Image


def open_lmdb(path):
    path = os.path.expanduser(path)
    env = lmdb.open(path, max_readers=32, readonly=True, lock=False,
                    readahead=False, meminit=False)
    txn = env.begin(write=False)
    num_samples = int(txn.get(b'num-samples').decode())
    return env, txn, num_samples


def get_image_bytes(txn, idx):
    img_key = 'image-%09d'.encode() % idx
    return txn.get(img_key)


def get_label(txn, idx):
    label_key = 'label-%09d'.encode() % idx
    val = txn.get(label_key)
    return val.decode('utf-8') if val else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lmdb_a', help='Path to first LMDB')
    parser.add_argument('lmdb_b', help='Path to second LMDB')
    parser.add_argument('--output_dir', default='./output/analysis/compare_lmdb',
                        help='Directory to save diff samples')
    parser.add_argument('--num_check', type=int, default=10,
                        help='Number of images to check for identity')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of diff samples to save')
    args = parser.parse_args()

    env_a, txn_a, n_a = open_lmdb(args.lmdb_a)
    env_b, txn_b, n_b = open_lmdb(args.lmdb_b)

    print(f'LMDB A: {args.lmdb_a} ({n_a} samples)')
    print(f'LMDB B: {args.lmdb_b} ({n_b} samples)')

    if n_a != n_b:
        print(f'[WARNING] Sample count mismatch: {n_a} vs {n_b}')
        print('Aborting.')
        return

    # Step 1: Check first N images are identical
    check_n = min(args.num_check, n_a)
    print(f'\n--- Checking first {check_n} images are identical ---')
    all_same = True
    for i in range(1, check_n + 1):
        img_a = get_image_bytes(txn_a, i)
        img_b = get_image_bytes(txn_b, i)
        if img_a != img_b:
            print(f'  Image {i}: DIFFERENT')
            all_same = False
        else:
            print(f'  Image {i}: OK')

    if not all_same:
        print('\nImages differ — these are not the same dataset. Aborting.')
        return

    print('All checked images are identical. Proceeding to label comparison.\n')

    # Step 2: Compare all labels
    print('--- Comparing all labels ---')
    diff_indices = []
    for i in range(1, n_a + 1):
        label_a = get_label(txn_a, i)
        label_b = get_label(txn_b, i)
        if label_a != label_b:
            diff_indices.append(i)

    diff_count = len(diff_indices)
    diff_ratio = diff_count / n_a * 100
    print(f'Total samples: {n_a}')
    print(f'Different labels: {diff_count} ({diff_ratio:.2f}%)')
    print(f'Same labels: {n_a - diff_count} ({100 - diff_ratio:.2f}%)')

    if diff_count == 0:
        print('\nNo differences found.')
        return

    # Step 3: Save random diff samples
    sample_n = min(args.num_samples, diff_count)
    sampled = random.sample(diff_indices, sample_n)
    sampled.sort()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append(f'LMDB A: {args.lmdb_a}')
    lines.append(f'LMDB B: {args.lmdb_b}')
    lines.append(f'Total: {n_a}, Diff: {diff_count} ({diff_ratio:.2f}%)')
    lines.append('')
    lines.append(f'{"idx":<12} {"label_A":<30} {"label_B":<30}')
    lines.append('-' * 72)

    for i, idx in enumerate(sampled):
        label_a = get_label(txn_a, idx)
        label_b = get_label(txn_b, idx)
        lines.append(f'{idx:<12} {label_a:<30} {label_b:<30}')

        # Save image
        img_bytes = get_image_bytes(txn_a, idx)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(output_dir, f'{i:02d}_idx{idx}.png'))

    report = '\n'.join(lines)
    print(f'\n--- Diff samples ({sample_n}) ---')
    print(report)

    report_path = os.path.join(output_dir, 'diff_labels.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f'\nSaved {sample_n} images + diff_labels.txt to {output_dir}')

    env_a.close()
    env_b.close()


if __name__ == '__main__':
    main()
