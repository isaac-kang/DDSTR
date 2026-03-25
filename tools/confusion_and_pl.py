#!/usr/bin/env python3
"""CCD: Confusion-aware Class Decomposition.

Step 1: Build character-level confusion matrix from pretrained model on train data.
        Extract confused classes per character and create extended class mapping.
Step 2: Perform Pseudo-Labeling (PL) on train datasets using the confusion mapping.
        Writes new LMDBs with Unicode-variant labels for confused characters.

Uses OpenOCR model loading (Config + build_model + load_ckpt).
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import lmdb as lmdb_lib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__)))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from engine.config import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from utils.ckpt import load_ckpt


# ==================== Unicode Variant Pool ====================

UNICODE_VARIANTS = {
    # Digits — Superscripts > Subscripts > Enclosed Alphanumerics > Math
    '0': '⁰₀⓪𝟎𝟘𝟢𝟬𝟶',
    '1': '¹₁①𝟏𝟙𝟣𝟭𝟷',
    '2': '²₂②𝟐𝟚𝟤𝟮𝟸',
    '3': '³₃③𝟑𝟛𝟥𝟯𝟹',
    '4': '⁴₄④𝟒𝟜𝟦𝟰𝟺',
    '5': '⁵₅⑤𝟓𝟝𝟧𝟱𝟻',
    '6': '⁶₆⑥𝟔𝟞𝟨𝟲𝟼',
    '7': '⁷₇⑦𝟕𝟟𝟩𝟳𝟽',
    '8': '⁸₈⑧𝟖𝟠𝟪𝟴𝟾',
    '9': '⁹₉⑨𝟗𝟡𝟫𝟵𝟿',
    # Lowercase — Latin-1 Supp > Ext-A > Ext-Additional > Ext-B > IPA
    'a': 'àáâãäåāăąǎǟǡȁȃạảấầẩẫậắằẳẵặ',
    'b': 'ḃḅḇƀƃɓᵬᶀ',
    'c': 'çćĉċčḉｃ𝐜',
    'd': 'ďḋḍḏḑḓđ',
    'e': 'èéêëēĕėęěȅȇẹẻẽếềểễệ',
    'f': 'ḟƒᵮᶂ',
    'g': 'ĝğġģǧǵḡ',
    'h': 'ĥȟḣḥḧḩḫẖħ',
    'i': 'ìíîïĩīĭįǐȉȋịỉĩḭ',
    'j': 'ĵǰɉʝ',
    'k': 'ķǩḱḳḵƙ',
    'l': 'ĺļľḷḹḻḽŀł',
    'm': 'ḿṁṃ',
    'n': 'ñńņňǹṅṇṉṋ',
    'o': 'òóôõöōŏőơǒǫǭȍȏọỏốồổỗộớờởỡợ',
    'p': 'ṕṗƥᵽᶈ',
    'q': 'ɋʠ',
    'r': 'ŕŗřȑȓṙṛṝṟ',
    's': 'śŝşšșṡṣṥṧṩ',
    't': 'ţťțṫṭṯṱẗ',
    'u': 'ùúûüũūŭůűųưǔǖǘǚǜȕȗụủứừửữự',
    'v': 'ṽṿʋᶌｖ𝐯',
    'w': 'ŵẁẃẅẇẉẘ',
    'x': 'ẋẍᶍ',
    'y': 'ýÿŷȳẏẙỳỵỷỹ',
    'z': 'źżžẑẓẕ',
    # Uppercase — Latin-1 Supp > Ext-A > Ext-Additional > Ext-B
    'A': 'ÀÁÂÃÄÅĀĂĄǍǞǠȀȂẠẢẤẦẨẪẬẮẰẲẴẶ',
    'B': 'ḂḄḆƁƂɃ',
    'C': 'ÇĆĈĊČḈ',
    'D': 'ĎḊḌḎḐḒĐ',
    'E': 'ÈÉÊËĒĔĖĘĚȄȆẸẺẼẾỀỂỄỆ',
    'F': 'ḞƑＦ𝐅',
    'G': 'ĜĞĠĢǦǴḠ',
    'H': 'ĤȞḢḤḦḨḪ',
    'I': 'ÌÍÎÏĨĪĬĮİǏȈȊỊỈĨḬ',
    'J': 'ĴɈ',
    'K': 'ĶǨḰḲḴƘ',
    'L': 'ĹĻĽḶḸḺḼĿŁ',
    'M': 'ḾṀṂＭ𝐌𝑀',
    'N': 'ÑŃŅŇǸṄṆṈṊ',
    'O': 'ÒÓÔÕÖŌŎŐƠǑǪǬȌȎỌỎỐỒỔỖỘỚỜỞỠỢ',
    'P': 'ṔṖƤＰ',
    'Q': 'Ɋ',
    'R': 'ŔŖŘȐȒṘṚṜṞ',
    'S': 'ŚŜŞŠȘṠṢṤṦṨ',
    'T': 'ŢŤȚṪṬṮṰ',
    'U': 'ÙÚÛÜŨŪŬŮŰŲƯǓǕǗǙǛȔȖỤỦỨỪỬỮỰ',
    'V': 'ṼṾƲＶ𝐕',
    'W': 'ŴẀẂẄẆẈ',
    'X': 'ẊẌ',
    'Y': 'ÝŶŸȲẎỲỴỶỸ',
    'Z': 'ŹŻŽẐẒẔ',
}


# ==================== Dataset =====================

class InferenceLMDBDataset(Dataset):
    """Loads images+labels from an LMDB, applies transforms, tracks raw LMDB indices."""

    def __init__(self, lmdb_dir, ops, post_process, max_text_length):
        self.lmdb_dir = str(lmdb_dir)
        self.ops = ops
        self.max_text_length = max_text_length

        env = lmdb_lib.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            self.num_samples = int(txn.get(b'num-samples').decode())
        env.close()

        # Build filtered index list
        self.filtered_indices = []  # 1-based LMDB indices
        self.labels = []
        env = lmdb_lib.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for idx in range(1, self.num_samples + 1):
                label_key = f'label-{idx:09d}'.encode()
                label = txn.get(label_key)
                if label is None:
                    continue
                label = label.decode('utf-8').strip()
                if len(label) > max_text_length or len(label) == 0:
                    continue
                # Check all chars are in the charset
                valid = True
                for ch in label:
                    if ch not in post_process.dict:
                        valid = False
                        break
                if valid:
                    self.filtered_indices.append(idx)
                    self.labels.append(label)
        env.close()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, i):
        lmdb_idx = self.filtered_indices[i]
        env = lmdb_lib.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            img_key = f'image-{lmdb_idx:09d}'.encode()
            imgbuf = txn.get(img_key)
        env.close()

        label = self.labels[i]
        data = {'image': imgbuf, 'label': label}
        data = transform(data, self.ops)
        if data is None:
            return None
        img = data[0]
        return img, label, lmdb_idx  # lmdb_idx is 1-based


def collate_fn(batch):
    """Filter out None samples and collate."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, lmdb_indices = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = torch.zeros(len(imgs), imgs[0].shape[0], max_h, max_w)
    for i, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        padded[i, :, :img.shape[1], :img.shape[2]] = img
    return padded, list(labels), list(lmdb_indices)


# ==================== Alignment & Confusion ====================

def needleman_wunsch_align(s1, s2, match_score=1, mismatch_score=-1, gap_score=-1):
    """Needleman-Wunsch alignment for two strings.
    Returns list of (char_from_s1_or_None, char_from_s2_or_None) pairs.
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_score
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_score
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match_score if s1[i - 1] == s2[j - 1] else mismatch_score
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap_score,
                dp[i][j - 1] + gap_score,
            )
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match_score if s1[i - 1] == s2[j - 1] else mismatch_score
            if dp[i][j] == dp[i - 1][j - 1] + score:
                alignment.append((s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap_score:
            alignment.append((s1[i - 1], None))
            i -= 1
        else:
            alignment.append((None, s2[j - 1]))
            j -= 1
    alignment.reverse()
    return alignment


def build_unicode_mapping(confusion_detail):
    """Assign a unique Unicode character to each extended class."""
    ext_to_unicode = {}
    unicode_to_ext = {}
    used = set()

    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        pool = UNICODE_VARIANTS.get(ch, '')
        idx = 0
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            while idx < len(pool) and pool[idx] in used:
                idx += 1
            if idx < len(pool):
                uni_ch = pool[idx]
                ext_to_unicode[ext_name] = uni_ch
                unicode_to_ext[uni_ch] = ext_name
                used.add(uni_ch)
                idx += 1
            else:
                ext_to_unicode[ext_name] = f'[{ext_name}]'
                print(f'WARNING: No Unicode variant left for {ext_name} (base={ch})')

    return ext_to_unicode, unicode_to_ext


def build_confusion_matrix(model, cfg, post_process, data_root, train_dirs, device,
                           batch_size=512, num_workers=4):
    """Run inference on train datasets and build character-level confusion matrix.
    Uses Needleman-Wunsch alignment to handle length mismatches.
    """
    confusion = defaultdict(lambda: defaultdict(int))

    # Build image-only transforms from config
    train_cfg = cfg.get('Train', cfg.get('Eval', {}))
    dataset_cfg = train_cfg.get('dataset', {})
    transforms_cfg = dataset_cfg.get('transforms', [])
    img_transforms = []
    for t in transforms_cfg:
        if isinstance(t, dict):
            name = list(t.keys())[0]
            if 'Encode' not in name and 'KeepKeys' not in name:
                img_transforms.append(t)
    img_transforms.append({'KeepKeys': {'keep_keys': ['image']}})
    ops = create_operators(img_transforms, cfg['Global'])

    max_text_length = cfg['Global'].get('max_text_length', 25)

    for train_dir in train_dirs:
        train_root = Path(data_root) / train_dir
        if (train_root / 'data.mdb').exists():
            lmdb_dirs = [train_root]
        else:
            lmdb_dirs = sorted(p.parent for p in train_root.rglob('data.mdb'))

        if not lmdb_dirs:
            print(f'WARNING: No LMDB found under {train_root}')
            continue

        for lmdb_dir in lmdb_dirs:
            ds_name = str(lmdb_dir.relative_to(Path(data_root)))
            dataset = InferenceLMDBDataset(lmdb_dir, ops, post_process, max_text_length)
            if len(dataset) == 0:
                print(f'  Skipping {ds_name}: no valid samples')
                continue

            dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True)
            print(f'Processing {ds_name} ({len(dataset)} samples)...')

            for batch in tqdm(dataloader, desc=ds_name):
                if batch is None:
                    continue
                imgs, labels, _ = batch
                imgs = imgs.to(device)
                logits = model(imgs)

                # Decode predictions
                if isinstance(logits, (list, tuple)):
                    pred_results = post_process(logits)
                else:
                    probs = logits.softmax(-1)
                    probs_np = probs.detach().cpu().numpy()
                    pred_results = post_process(probs_np)
                preds_text = [r[0] for r in pred_results]

                for pred, gt in zip(preds_text, labels):
                    aligned = needleman_wunsch_align(gt, pred)
                    for gt_ch, pred_ch in aligned:
                        if gt_ch is not None and pred_ch is not None:
                            if gt_ch.lower() == pred_ch.lower():
                                confusion[gt_ch][gt_ch] += 1
                            else:
                                confusion[gt_ch][pred_ch] += 1

    return confusion


def extract_confusions(confusion, charset, min_rate=0.001):
    """For each character in charset, find confused characters with rate >= min_rate."""
    mapping = {}
    extended_classes = {}
    confusion_detail = {}

    for ch in sorted(charset):
        if ch not in confusion:
            continue
        correct_count = confusion[ch].get(ch, 0)
        total = sum(confusion[ch].values())
        if total == 0:
            continue

        confused = {k: v for k, v in confusion[ch].items() if k != ch}
        if not confused:
            continue

        sorted_confused = sorted(confused.items(), key=lambda x: (-x[1], x[0]))
        filtered = [(c, cnt) for c, cnt in sorted_confused if cnt / total >= min_rate]

        if not filtered:
            continue

        mapping[ch] = [c for c, _ in filtered]
        extended_classes[ch] = [ch] + [f'{ch}_{i+1}' for i in range(len(filtered))]

        confusion_detail[ch] = {
            'correct': correct_count,
            'total': total,
            'accuracy': correct_count / total,
            'confused': [{'char': c, 'count': cnt, 'rate': cnt / total} for c, cnt in filtered],
            'extended_classes': extended_classes[ch],
            'extended_class_mapping': {
                extended_classes[ch][0]: ch,
                **{extended_classes[ch][i+1]: filtered[i][0] for i in range(len(filtered))}
            }
        }

    return mapping, extended_classes, confusion_detail


def _build_confusion_map(confusion_detail, ext_to_unicode):
    """Build reverse lookup: for each gt_char, {confused_char -> unicode_char}."""
    confusion_map = {}
    for ch, detail in confusion_detail.items():
        ext_mapping = detail['extended_class_mapping']
        reverse = {}
        for ext_name, actual_char in ext_mapping.items():
            if ext_name != ch:
                uni_ch = ext_to_unicode.get(ext_name, ext_name)
                reverse[actual_char] = uni_ch
        confusion_map[ch] = reverse
    return confusion_map


def _apply_pl(gt, pred, confusion_map):
    """Apply PL rule to a single (gt, pred) pair. Returns PL string."""
    aligned = needleman_wunsch_align(gt, pred)
    pl_chars = []
    for gt_ch, pred_ch in aligned:
        if gt_ch is None:
            continue
        if pred_ch is None:
            pl_chars.append(gt_ch)
        elif gt_ch.lower() == pred_ch.lower():
            pl_chars.append(gt_ch)
        elif gt_ch in confusion_map and pred_ch in confusion_map[gt_ch]:
            pl_chars.append(confusion_map[gt_ch][pred_ch])
        else:
            pl_chars.append(gt_ch)
    return ''.join(pl_chars)


def perform_pl(model, cfg, post_process, lmdb_dir, ds_name, confusion_detail,
               ext_to_unicode, device, lmdb_output_path, batch_size=128, num_workers=4):
    """Perform pseudo-labeling on a dataset and write LMDB with PL labels.

    Returns dict with stats: total_samples, seq_changed, total_chars, chars_extended.
    """
    confusion_map = _build_confusion_map(confusion_detail, ext_to_unicode)
    ext_unicode_set = set(ext_to_unicode.values())

    # Build image-only transforms
    train_cfg = cfg.get('Train', cfg.get('Eval', {}))
    dataset_cfg = train_cfg.get('dataset', {})
    transforms_cfg = dataset_cfg.get('transforms', [])
    img_transforms = []
    for t in transforms_cfg:
        if isinstance(t, dict):
            name = list(t.keys())[0]
            if 'Encode' not in name and 'KeepKeys' not in name:
                img_transforms.append(t)
    img_transforms.append({'KeepKeys': {'keep_keys': ['image']}})
    ops = create_operators(img_transforms, cfg['Global'])

    max_text_length = cfg['Global'].get('max_text_length', 25)

    dataset = InferenceLMDBDataset(lmdb_dir, ops, post_process, max_text_length)
    if len(dataset) == 0:
        print(f'  Skipping {ds_name}: no valid samples')
        return []

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True)

    results = []
    for batch in tqdm(dataloader, desc=f'{ds_name} PL'):
        if batch is None:
            continue
        imgs, labels, lmdb_indices = batch
        imgs = imgs.to(device)
        logits = model(imgs)

        if isinstance(logits, (list, tuple)):
            pred_results = post_process(logits)
        else:
            probs = logits.softmax(-1)
            probs_np = probs.detach().cpu().numpy()
            pred_results = post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        for pred, gt, lmdb_idx in zip(preds_text, labels, lmdb_indices):
            pl_unicode = _apply_pl(gt, pred, confusion_map)
            results.append({
                'gt': gt,
                'pred': pred,
                'pl': pl_unicode,
                'lmdb_idx': lmdb_idx,  # 1-based
            })

    # Stats
    n_changed = sum(1 for r in results if r['gt'] != r['pred'])
    n_pl_applied = sum(1 for r in results if r['gt'] != r['pl'])
    total_chars = sum(len(r['gt']) for r in results)
    extended_chars = 0
    for r in results:
        extended_chars += sum(1 for c in r['pl'] if c in ext_unicode_set)
    if total_chars > 0:
        print(f'  Total: {len(results)}, wrong pred: {n_changed}, '
              f'PL applied (seq): {n_pl_applied} ({n_pl_applied/len(results)*100:.2f}%), '
              f'extended chars: {extended_chars}/{total_chars} ({extended_chars/total_chars*100:.3f}%)')
    else:
        print(f'  Total: {len(results)}, wrong pred: {n_changed}, PL applied: {n_pl_applied}')

    # Write LMDB with PL labels
    if lmdb_output_path:
        lmdb_output_path = Path(lmdb_output_path)
        lmdb_output_path.mkdir(parents=True, exist_ok=True)

        src_env = lmdb_lib.open(str(lmdb_dir), readonly=True, lock=False)
        src_mdb = Path(lmdb_dir) / 'data.mdb'
        map_size = max(int(src_mdb.stat().st_size * 1.5), 1024 * 1024 * 100)
        dst_env = lmdb_lib.open(str(lmdb_output_path), map_size=map_size)

        with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
            dst_txn.put('num-samples'.encode(), str(len(results)).encode())
            for out_idx, r in enumerate(results, start=1):
                src_img_key = f'image-{r["lmdb_idx"]:09d}'.encode()
                img_data = src_txn.get(src_img_key)
                dst_txn.put(f'image-{out_idx:09d}'.encode(), img_data)
                dst_txn.put(f'label-{out_idx:09d}'.encode(), r['pl'].encode())

        dst_env.close()
        src_env.close()
        print(f'  PL LMDB saved to {lmdb_output_path}')

    return {
        'total_samples': len(results),
        'seq_changed': n_pl_applied,
        'total_chars': total_chars,
        'chars_extended': extended_chars,
    }


# ==================== Model Loading ====================

def load_model(config_path, checkpoint_path, device):
    """Load an OpenOCR model from config and checkpoint."""
    cfg = Config(config_path).cfg

    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    model = build_model(cfg['Architecture'])

    if checkpoint_path:
        cfg['Global']['pretrained_model'] = checkpoint_path
    load_ckpt(model, cfg)

    model.eval().to(device)
    return model, cfg, post_process


# ==================== Main ====================

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='CCD: Confusion-aware Class Decomposition')
    parser.add_argument('--config', '-c', required=True, help='Model config YAML')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint path')
    parser.add_argument('--data_root', default='~/data/STR/openocr',
                        help='Root dir containing training LMDBs')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default=None,
                        help='Output dir for confusion matrix/mapping (default: ~/data/STR/ddstr/CCD/{model_type}/)')
    parser.add_argument('--min_rate', type=float, default=0.001,
                        help='Minimum confusion rate threshold (default: 0.001 = 0.1%%)')
    parser.add_argument('--train_dirs', nargs='+',
                        default=['Union14M-L-LMDB-Filtered/filter_train_challenging',
                                 'Union14M-L-LMDB-Filtered/filter_train_hard',
                                 'Union14M-L-LMDB-Filtered/filter_train_medium',
                                 'Union14M-L-LMDB-Filtered/filter_train_normal',
                                 'Union14M-L-LMDB-Filtered/filter_train_easy'],
                        help='Train dirs relative to data_root for confusion matrix')
    parser.add_argument('--pl_datasets', nargs='+', default=None,
                        help='Datasets for PL (relative to data_root). Defaults to --train_dirs.')
    parser.add_argument('--pl_output_root', default=None,
                        help='Root dir for decomposed LMDB output')
    parser.add_argument('--skip_pl', action='store_true',
                        help='Skip PL (Step 2), only build confusion matrix')
    parser.add_argument('--model_type', default=None,
                        help='Model type name for output path (auto-detected from config if omitted)')
    args = parser.parse_args()

    args.data_root = str(Path(args.data_root).expanduser().resolve())

    # Load model
    print(f'Loading model from {args.config}...')
    model, cfg, post_process = load_model(args.config, args.checkpoint, args.device)

    # Detect model type for default paths
    if args.model_type is None:
        decoder_name = cfg['Architecture']['Decoder']['name']
        if 'CTC' in decoder_name or 'RCTC' in decoder_name:
            args.model_type = 'svtrv2_ctc'
        elif 'IGTR' in decoder_name:
            args.model_type = 'igtr_ar'
        elif 'PARSeq' in decoder_name:
            args.model_type = 'parseq_ar'
        elif 'MDiff' in decoder_name:
            args.model_type = 'mdiff_cmlm'
        else:
            args.model_type = 'unknown'

    if args.output_dir is None:
        args.output_dir = str(Path.home() / 'data' / 'STR' / 'ddstr' / 'CCD' / args.model_type)
    else:
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    if args.pl_output_root is None:
        args.pl_output_root = args.output_dir
    else:
        args.pl_output_root = str(Path(args.pl_output_root).expanduser().resolve())

    if args.pl_datasets is None:
        args.pl_datasets = args.train_dirs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get charset from post_process
    # Exclude special tokens (blank, eos, bos, pad)
    charset = set()
    for ch in post_process.dict.keys():
        if ch not in ('blank', 'sos', 'eos') and len(ch) == 1:
            charset.add(ch)
    charset = sorted(charset)
    print(f'Charset size: {len(charset)}')

    # ==================== Step 1: Confusion Matrix ====================
    print('\n' + '=' * 60)
    print(f'Step 1: Building confusion matrix ({", ".join(args.train_dirs)})')
    print('=' * 60)

    confusion = build_confusion_matrix(
        model, cfg, post_process, args.data_root, args.train_dirs,
        args.device, args.batch_size, args.num_workers,
    )

    # Save raw confusion matrix as numpy
    chars = sorted(charset)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    cm = np.zeros((n, n), dtype=np.int64)
    for gt_ch, preds in confusion.items():
        if gt_ch not in char_to_idx:
            continue
        for pred_ch, count in preds.items():
            if pred_ch not in char_to_idx:
                continue
            cm[char_to_idx[gt_ch], char_to_idx[pred_ch]] = count

    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Save confusion matrix as CSV
    csv_path = output_dir / 'confusion_matrix.csv'
    with open(csv_path, 'w') as f:
        f.write('gt\\pred,' + ','.join(chars) + '\n')
        for i, ch in enumerate(chars):
            f.write(ch + ',' + ','.join(str(cm[i, j]) for j in range(n)) + '\n')
    print(f'Confusion matrix saved to {csv_path}')

    # Extract confusions
    print(f'\nConfusion rate threshold: {args.min_rate*100:.1f}%')
    mapping, extended_classes, confusion_detail = extract_confusions(confusion, charset, min_rate=args.min_rate)

    # Save confusion mapping
    mapping_path = output_dir / 'confusion_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(confusion_detail, f, indent=2, ensure_ascii=False)
    print(f'Confusion mapping saved to {mapping_path}')

    # Build Unicode mapping
    ext_to_unicode, unicode_to_ext = build_unicode_mapping(confusion_detail)

    # Save Unicode mapping
    unicode_mapping_path = output_dir / 'unicode_mapping.json'
    mapping_data = {}
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            mapping_data[ext_name] = {
                'unicode': uni_ch,
                'codepoint': f'U+{ord(uni_ch):04X}' if len(uni_ch) == 1 else '?',
                'unicode_name': unicodedata.name(uni_ch, '?') if len(uni_ch) == 1 else '?',
                'base_char': ch,
                'confused_with': t['char'],
            }
    with open(unicode_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f'Unicode mapping saved to {unicode_mapping_path}')

    # Save extended class summary
    summary_path = output_dir / 'extended_classes.txt'
    total_extended = 0
    summary_lines = []
    summary_lines.append(f'Extended Class Summary (threshold >= {args.min_rate*100:.1f}%)')
    summary_lines.append('=' * 100)
    summary_lines.append('')
    summary_lines.append(f'{"GT":<5} {"Acc%":<7} {"ExtClass":<10} {"Unicode":<4} {"Codepoint":<10} {"->Confused":<12} {"Count":<7} {"Rate%":<7}')
    summary_lines.append('-' * 100)
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        acc = d['accuracy'] * 100
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            codepoint = f'U+{ord(uni_ch):04X}' if len(uni_ch) == 1 else '?'
            total_extended += 1
            prefix = f'{ch:<5} {acc:<7.1f}' if i == 0 else f'{"":5} {"":7}'
            summary_lines.append(f'{prefix} {ext_name:<10} {uni_ch:<4} {codepoint:<10} -> {t["char"]:<10} {t["count"]:<7} {t["rate"]*100:<7.2f}')
    summary_lines.append('-' * 100)
    summary_lines.append('')
    summary_lines.append(f'Original charset size: {len(charset)}')
    summary_lines.append(f'Characters with extended classes: {len(confusion_detail)}')
    summary_lines.append(f'Total extended classes added: {total_extended}')
    summary_lines.append(f'New total class count: {len(charset) + total_extended}')
    summary_lines.append('')
    all_ext = []
    for ch in sorted(confusion_detail.keys()):
        d = confusion_detail[ch]
        for i, t in enumerate(d['confused']):
            ext_name = d['extended_classes'][i + 1]
            uni_ch = ext_to_unicode.get(ext_name, '?')
            all_ext.append(f'{ext_name}={uni_ch}')
    summary_lines.append('All extended classes (name -> unicode):')
    summary_lines.append(', '.join(all_ext))
    summary_text = '\n'.join(summary_lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f'Extended class summary saved to {summary_path}')
    print(f'\n{summary_text}')

    # ==================== Step 2: PL ====================
    if args.skip_pl:
        print('\nSkipping Step 2 (PL dataset generation).')
    else:
        print('\n' + '=' * 60)
        print('Step 2: Pseudo-Labeling with confusion decomposition')
        print('=' * 60)

        data_root = Path(args.data_root)
        pl_output_root = Path(args.pl_output_root)

        agg_samples = 0
        agg_seq_changed = 0
        agg_chars = 0
        agg_chars_ext = 0
        per_dataset_stats = []

        for pl_ds in args.pl_datasets:
            ds_abs = data_root / pl_ds
            if (ds_abs / 'data.mdb').exists():
                sub_lmdbs = [ds_abs]
            else:
                sub_lmdbs = sorted(p.parent for p in ds_abs.rglob('data.mdb'))

            if not sub_lmdbs:
                print(f'\nWARNING: No LMDB found under {ds_abs}, skipping.')
                continue

            print(f'\nFound {len(sub_lmdbs)} LMDB(s) under {pl_ds}')
            for lmdb_dir in sub_lmdbs:
                rel = str(lmdb_dir.relative_to(data_root))
                lmdb_out = str(pl_output_root / rel)
                print(f'\n  Dataset: {rel}')
                ds_stats = perform_pl(model, cfg, post_process, lmdb_dir, rel,
                                      confusion_detail, ext_to_unicode, args.device,
                                      lmdb_out, args.batch_size, args.num_workers)
                agg_samples += ds_stats['total_samples']
                agg_seq_changed += ds_stats['seq_changed']
                agg_chars += ds_stats['total_chars']
                agg_chars_ext += ds_stats['chars_extended']
                per_dataset_stats.append({'dataset': rel, **ds_stats})

        # Save PL stats
        pl_stats = {
            'total_samples': agg_samples,
            'seq_changed': agg_seq_changed,
            'seq_change_ratio': round(agg_seq_changed / agg_samples, 6) if agg_samples > 0 else 0.0,
            'total_chars': agg_chars,
            'chars_extended': agg_chars_ext,
            'char_extend_ratio': round(agg_chars_ext / agg_chars, 6) if agg_chars > 0 else 0.0,
            'per_dataset': per_dataset_stats,
        }
        pl_stats_path = output_dir / 'pl_stats.json'
        with open(pl_stats_path, 'w') as f:
            json.dump(pl_stats, f, indent=2)
        print(f'\nPL stats saved to {pl_stats_path}')
        if agg_samples > 0:
            print(f'  Sequence-level: {agg_seq_changed}/{agg_samples} '
                  f'({agg_seq_changed/agg_samples*100:.2f}%) sequences changed')
        if agg_chars > 0:
            print(f'  Char-level: {agg_chars_ext}/{agg_chars} '
                  f'({agg_chars_ext/agg_chars*100:.3f}%) chars extended')

    print(f'\nAll outputs saved to {output_dir}/')


if __name__ == '__main__':
    main()
