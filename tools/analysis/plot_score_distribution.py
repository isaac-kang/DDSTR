"""Compute STR/LLM scores for PL error samples and plot distributions.

Reads PL_oracle.csv, loads images from eval LMDBs,
runs STR model for conf_str and LLM for conf_llm, then plots distributions.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/analysis/plot_score_distribution.py
"""

import argparse
import math
import csv
import io
import os
import sys
from pathlib import Path

import lmdb as lmdb_lib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine.config import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from utils.ckpt import load_ckpt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_entries(path):
    """Load PL-labeled entries from CSV or Excel.

    Supports:
      - Old CSV format: dataset_name, image_index, pred, gt, PL
      - New Excel format: dataset, lmdb_index, pred, gt, PL_label (1/2/3)
    """
    entries = []
    if path.endswith('.xlsx'):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(min_row=1, values_only=True))
        headers = [str(h).strip().lower() if h else '' for h in rows[0]]
        for row in rows[1:]:
            vals = {headers[i]: row[i] for i in range(len(headers)) if i < len(row)}
            # Skip instruction rows
            ds = vals.get('dataset', vals.get('dataset_name', ''))
            if not ds or str(ds).startswith('1='):
                continue
            idx = vals.get('lmdb_index', vals.get('image_index', 0))
            if idx is None or idx == '':
                continue
            pred = str(vals.get('pred', ''))
            gt = str(vals.get('gt', ''))
            pl_label = vals.get('pl_label', vals.get('pl', ''))
            if pl_label is None:
                pl_label = ''
            pl_label = str(pl_label).strip()
            # Convert numeric PL_label to text
            if pl_label == '1':
                pl = pred
            elif pl_label == '2':
                pl = gt
            elif pl_label == '3' or pl_label.lower() in ('other', 'illegible'):
                pl = ''  # treated as blank/illegible
            else:
                pl = pl_label  # keep as-is (might be empty)
            entries.append({
                'dataset_name': str(ds),
                'image_index': int(idx),
                'pred': pred,
                'gt': gt,
                'PL': pl,
            })
        wb.close()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            has_lmdb_index = 'lmdb_index' in reader.fieldnames
            for row in reader:
                ds = row.get('dataset_name', row.get('dataset', ''))
                if has_lmdb_index:
                    idx = int(row['lmdb_index'])  # already 1-based
                else:
                    idx = int(row.get('image_index', 0)) + 1  # 0-based -> 1-based
                pl = row.get('PL', row.get('PL_label', ''))
                pred = row['pred']
                gt = row['gt']
                if pl and pl in ('1', '2', '3'):
                    if pl == '1':
                        pl = pred
                    elif pl == '2':
                        pl = gt
                    else:
                        pl = ''
                entries.append({
                    'dataset_name': ds,
                    'image_index': idx,  # always 1-based
                    'pred': pred,
                    'gt': gt,
                    'PL': pl if pl else '',
                })
    return entries


def categorize(entry):
    pl = entry['PL']
    if pl == '' or pl is None:
        return 'other'
    if pl == entry['pred']:
        return 'PL=pred'
    if pl == entry['gt']:
        return 'PL=gt'
    return 'PL=other'


def load_image_from_lmdb(eval_root, dataset_name, image_index):
    """Load raw image bytes from eval LMDB. image_index is 1-based (LMDB native)."""
    lmdb_dir = os.path.join(eval_root, dataset_name)
    lmdb_index = image_index
    env = lmdb_lib.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        img_key = f'image-{lmdb_index:09d}'.encode()
        imgbuf = txn.get(img_key)
        imgbuf = bytes(imgbuf)
    env.close()
    return imgbuf


# ===================== STR Scoring =====================

def compute_str_scores(entries, config_path, checkpoint_path, eval_root, device, opt_overrides=None):
    """Compute pred and GT scores for each entry.

    Returns (pred_score_list, gt_score_list, per_char_list).
    Scores are in log-prob space (mean log prob).
    """
    from denoise.extract_error_info import detect_model_type, get_scorer

    config = Config(config_path)
    if opt_overrides:
        config.merge_dict(opt_overrides)
    cfg = config.cfg
    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_model(cfg['Architecture'])
    if not cfg['Global'].get('pretrained_model'):
        cfg['Global']['pretrained_model'] = checkpoint_path
    load_ckpt(model, cfg)
    model.eval().to(device)

    model_type = detect_model_type(cfg)
    scorer = get_scorer(model_type, post_process, device)

    eval_cfg = cfg.get('Eval', cfg.get('Train', {}))
    ds_cfg = eval_cfg.get('dataset', {})
    padding = ds_cfg.get('padding', False)
    base_shape = ds_cfg.get('base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
    base_h = ds_cfg.get('base_h', 32)

    # Build transforms for decoding image (same as RatioDataSetTVResize)
    transforms_cfg = ds_cfg.get('transforms', [])
    img_transforms = []
    for t in transforms_cfg:
        if isinstance(t, dict):
            name = list(t.keys())[0]
            if 'Encode' not in name and 'KeepKeys' not in name:
                img_transforms.append(t)
    img_transforms.append({'KeepKeys': {'keep_keys': ['image']}})
    decode_ops = create_operators(img_transforms, cfg['Global'])

    interpolation = T.InterpolationMode.BICUBIC
    img_normalize = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    pred_score_list = []
    gt_score_list = []
    per_char_list = []
    print(f'Computing STR scores ({len(entries)} samples)...')
    with torch.inference_mode():
        for i, entry in enumerate(entries):
            imgbuf = load_image_from_lmdb(eval_root, entry['dataset_name'], entry['image_index'])
            # Decode image (apply non-Encode, non-KeepKeys transforms)
            data = {'image': imgbuf, 'label': entry['gt']}
            data = transform(data, decode_ops)
            if data is None:
                pred_score_list.append(0.0)
                gt_score_list.append(0.0)
                per_char_list.append({'pred_text': '', 'chars': [], 'probs': []})
                continue
            pil_img = data[0]
            w, h = pil_img.size
            ratio = max(1, round(w / h))
            # Match RatioDataSetTVResize.resize_norm_img
            if ratio <= 4:
                imgW, imgH = base_shape[ratio - 1]
            else:
                imgW, imgH = base_h * ratio, base_h
            if not padding:
                resized_w = imgW
            else:
                r = w / float(h)
                import math
                resized_w = min(imgW, math.ceil(imgH * r))
            resized_image = TF.resize(pil_img, (imgH, resized_w), interpolation=interpolation)
            img = img_normalize(resized_image)
            if resized_w < imgW and padding:
                img = TF.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
            img = img.unsqueeze(0).to(device)

            preds_text, pred_scores, gt_scores = scorer.score(model, img, [entry['gt']])
            pred_score_list.append(pred_scores[0].item())
            gt_score_list.append(gt_scores[0].item())

            pred_text = preds_text[0]
            char_probs = scorer.last_char_probs if hasattr(scorer, 'last_char_probs') else []
            per_char_list.append({
                'pred_text': pred_text,
                'chars': list(pred_text),
                'probs': char_probs,
                'pred_ar_lp': getattr(scorer, 'last_pred_ar_lp', None),
                'pred_cloze_lp': getattr(scorer, 'last_pred_cloze_lp', None),
                'gt_ar_lp': getattr(scorer, 'last_gt_ar_lp', None),
                'gt_cloze_lp': getattr(scorer, 'last_gt_cloze_lp', None),
                'alignment_detail': getattr(scorer, 'last_alignment_detail', None),
            })

            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(entries)}')

    del model
    torch.cuda.empty_cache()
    return pred_score_list, gt_score_list, per_char_list


# ===================== LLM Scoring =====================

def build_prompt(pred, gt):
    return f"Which is more likely the true text, allowing for OCR mistakes (e.g., similar-looking characters)? 1) {pred} 2) {gt} Answer only 1 or 2."


def extract_llm_logodds(output):
    """Extract lo_llm = lp("1") - lp("2") from vLLM output."""
    logprobs_list = output.outputs[0].logprobs
    if not logprobs_list:
        return 0.0

    generated_text = output.outputs[0].text
    char_pos = None
    for i, ch in enumerate(generated_text):
        if ch in ("1", "2"):
            char_pos = i
            break
    if char_pos is None:
        return 0.0

    token_ids = output.outputs[0].token_ids
    answer_token_idx = None
    cum_len = 0
    for tok_idx, tid in enumerate(token_ids):
        if tok_idx >= len(logprobs_list):
            break
        token_logprobs = logprobs_list[tok_idx]
        if tid in token_logprobs:
            decoded = token_logprobs[tid].decoded_token
        else:
            decoded = next(iter(token_logprobs.values())).decoded_token
        if cum_len + len(decoded) > char_pos:
            answer_token_idx = tok_idx
            break
        cum_len += len(decoded)

    if answer_token_idx is None:
        return 0.0

    target_logprobs = logprobs_list[answer_token_idx]
    lp1, lp2 = None, None
    for token_id, logprob_obj in target_logprobs.items():
        decoded = logprob_obj.decoded_token
        if "1" in decoded and lp1 is None:
            lp1 = logprob_obj.logprob
        elif "2" in decoded and lp2 is None:
            lp2 = logprob_obj.logprob

    if lp1 is None and lp2 is None:
        return 0.0
    if lp1 is None:
        return -20.0
    if lp2 is None:
        return 20.0
    return lp1 - lp2


def compute_llm_scores(entries, llm_model_name, device):
    """Compute lo_llm = lp("1") - lp("2") for each entry using vLLM."""
    from vllm import LLM, SamplingParams

    print(f'Loading LLM: {llm_model_name}...')
    llm = LLM(model=llm_model_name, max_model_len=512, enforce_eager=True)
    sampling_params = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)

    conversations = []
    chat_kwargs = {}
    if "Qwen3" in llm_model_name:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    for entry in entries:
        prompt = build_prompt(entry['pred'], entry['gt'])
        conversations.append([{"role": "user", "content": prompt}])

    print(f'Computing LLM scores ({len(entries)} samples)...')
    outputs = llm.chat(conversations, sampling_params=sampling_params, **chat_kwargs)

    lo_llm_list = [extract_llm_logodds(output) for output in outputs]

    del llm
    torch.cuda.empty_cache()
    return lo_llm_list


# ===================== Plotting =====================

def plot_hist(ax, data_by_cat, cats, colors, bins, xlabel, title):
    data_list = [data_by_cat[c] for c in cats]
    ax.hist(data_list, bins=bins, stacked=True,
            color=[colors[c] for c in cats],
            label=cats, edgecolor='white', linewidth=0.5)
    ax.axvline(0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)


def plot_strip(ax, data_by_cat, cats, colors, bins, xlabel, title):
    np.random.seed(42)
    for idx, cat in enumerate(cats):
        subset = data_by_cat[cat]
        jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
        ax.scatter(subset, idx + jitter, c=colors[cat], alpha=0.6, s=30,
                   label=cat, edgecolors='white', linewidth=0.3)
    for b in bins:
        ax.axvline(b, color='gray', linewidth=0.5, alpha=0.4, linestyle='--')
    ax.axvline(0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=11)
    ax.set_title(title, fontsize=13)


def _save_per_char_xlsx(entries, conf_str, per_char_list, output_path):
    """Save per-character NAR probs to Excel."""
    xlsx_path = output_path.replace('.png', '_char_probs.xlsx')
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print(f'openpyxl not installed, skipping {xlsx_path}')
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'NAR Char Probs'

    headers = ['dataset', 'index', 'gt', 'NAR_pred', 'category',
               'str_score', 'pos', 'char', 'prob']
    header_font = Font(bold=True)
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h).font = header_font

    red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')
    row = 2
    for i, (entry, pc) in enumerate(zip(entries, per_char_list)):
        chars = pc['chars']
        probs = pc['probs']
        n_chars = max(len(chars), 1)
        for j in range(n_chars):
            if j == 0:
                ws.cell(row=row, column=1, value=entry['dataset_name'])
                ws.cell(row=row, column=2, value=entry['image_index'])
                ws.cell(row=row, column=3, value=entry['gt'])
                ws.cell(row=row, column=4, value=pc['pred_text'])
                ws.cell(row=row, column=5, value=entry.get('category', ''))
                ws.cell(row=row, column=6, value=round(float(conf_str[i]), 4))
            ws.cell(row=row, column=7, value=j)
            if j < len(chars):
                ws.cell(row=row, column=8, value=chars[j])
            if j < len(probs):
                c = ws.cell(row=row, column=9, value=round(probs[j], 4))
                if probs[j] < 0.5:
                    c.fill = red_fill
            row += 1

    os.makedirs(os.path.dirname(xlsx_path) or '.', exist_ok=True)
    wb.save(xlsx_path)
    print(f'Per-char probs saved to {xlsx_path}')


def _save_alignment_xlsx(entries, conf_str, pred_lp, gt_lp, per_char_list, output_path):
    """Save PD/MDM alignment detail to Excel."""
    xlsx_path = output_path.replace('.png', '_alignment.xlsx')
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print(f'openpyxl not installed, skipping {xlsx_path}')
        return

    # Check if any sample has alignment detail
    has_any = any(pc.get('alignment_detail') for pc in per_char_list)
    if not has_any:
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'PD Alignment'

    headers = ['dataset', 'index', 'pred', 'gt', 'category',
               'pred_lp', 'gt_lp', 'score_str', 'pred_cur',
               'match_type', 'op', 'pred_pos', 'pred_char', 'gt_idx', 'gt_char', 'gt_lp_pos']
    header_font = Font(bold=True)
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h).font = header_font

    green_fill = PatternFill(start_color='CCFFCC', end_color='CCFFCC', fill_type='solid')
    blue_fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
    gray_fill = PatternFill(start_color='DDDDDD', end_color='DDDDDD', fill_type='solid')
    red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')

    row = 2
    for i, (entry, pc) in enumerate(zip(entries, per_char_list)):
        detail = pc.get('alignment_detail')
        if not detail:
            continue

        n_rows = len(detail)
        for j, d in enumerate(detail):
            if j == 0:
                ws.cell(row=row, column=1, value=entry['dataset_name'])
                ws.cell(row=row, column=2, value=entry['image_index'])
                ws.cell(row=row, column=3, value=entry['pred'])
                ws.cell(row=row, column=4, value=entry['gt'])
                ws.cell(row=row, column=5, value=entry.get('category', ''))
                ws.cell(row=row, column=6, value=round(float(pred_lp[i]), 4))
                ws.cell(row=row, column=7, value=round(float(gt_lp[i]), 4))
                ws.cell(row=row, column=8, value=round(float(conf_str[i]), 4))
                ws.cell(row=row, column=9, value=pc.get('pred_text', ''))

            ws.cell(row=row, column=10, value=d['match_type'])
            ws.cell(row=row, column=11, value=d['op'])
            if d['pred_pos'] is not None:
                ws.cell(row=row, column=12, value=d['pred_pos'])
            ws.cell(row=row, column=13, value=d['pred_char'])
            if d['gt_idx'] is not None:
                ws.cell(row=row, column=14, value=d['gt_idx'])
            ws.cell(row=row, column=15, value=d['gt_char'])
            if d['gt_lp'] is not None:
                c = ws.cell(row=row, column=16, value=round(d['gt_lp'], 4))
                if d['gt_lp'] < -2.0:
                    c.fill = red_fill

            # Color by match type
            fill = None
            if d['match_type'] == 'match1':
                fill = green_fill
            elif d['match_type'] == 'match2':
                fill = blue_fill
            elif d['match_type'] == 'skip':
                fill = gray_fill
            if fill:
                for col in [10, 11, 12, 13, 14, 15]:
                    ws.cell(row=row, column=col).fill = fill

            row += 1
        row += 1  # blank row between samples

    os.makedirs(os.path.dirname(xlsx_path) or '.', exist_ok=True)
    wb.save(xlsx_path)
    print(f'Alignment detail saved to {xlsx_path}')


def main():
    MODEL_PRESETS = {
        'mdiff4str': {
            'config': 'configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml',
            'checkpoint': 'pretrained/mdiff4str_base/best.pth',
        },
        'parseq': {
            'config': 'configs/rec/parseq/svrtv2_parseq.yml',
            'checkpoint': 'pretrained/svtrv2_parseq/best.pth',
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None,
                        help='CSV or Excel (.xlsx) with PL labels (default: output/analysis/error_list_{model}.xlsx)')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_PRESETS.keys()),
                        help='Model preset (auto-sets config/checkpoint)')
    parser.add_argument('--config', '-c', default=None, help='STR model config')
    parser.add_argument('--checkpoint', default=None, help='STR model checkpoint')
    parser.add_argument('--eval_root', default='~/data/STR/openocr/evaluation',
                        help='Root dir for eval LMDBs')
    parser.add_argument('--llm_model', default='Qwen/Qwen3-32B-AWQ')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Blending weight: alpha*STR + (1-alpha)*LLM')
    parser.add_argument('--lo_clip', type=float, default=5.0,
                        help='Clip log-odds to [-lo_clip, lo_clip]')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold: score >= threshold -> pred, else -> gt')
    parser.add_argument('--str_cache', default=None,
                        help='STR scores cache file (default: tools/analysis/str_scores_cache_{model}.json)')
    parser.add_argument('--no_str_cache', action='store_true',
                        help='Ignore existing cache and recompute STR scores')
    parser.add_argument('--llm_cache', default=None,
                        help='LLM scores cache file (default: tools/analysis/llm_scores_cache_{model}.json)')
    parser.add_argument('--no_llm_cache', action='store_true',
                        help='Ignore existing cache and recompute LLM scores')
    parser.add_argument('--str_only', action='store_true',
                        help='Compute STR scores only (skip LLM)')
    parser.add_argument('--llm_only', action='store_true',
                        help='Compute LLM scores only (skip STR)')
    parser.add_argument('--output', default=None,
                        help='Output plot path (default: output/analysis/score_distribution_{model}.png)')
    parser.add_argument('-o', '--opt', nargs='*', default=[],
                        help='Override config options, e.g. -o Architecture.Decoder.decoding_mode=greedy')
    args = parser.parse_args()

    # Resolve model preset
    preset = MODEL_PRESETS[args.model]
    if args.config is None:
        args.config = preset['config']
    if args.checkpoint is None:
        args.checkpoint = preset['checkpoint']

    str_model_name = args.model
    if args.csv is None:
        args.csv = f'output/analysis/error_list_{str_model_name}.xlsx'
    if args.output is None:
        args.output = f'output/analysis/score_distribution_{str_model_name}.png'
    if args.str_cache is None:
        args.str_cache = f'tools/analysis/str_scores_cache_{str_model_name}.json'
    if args.llm_cache is None:
        llm_model_name = args.llm_model.replace('/', '-')
        args.llm_cache = f'tools/analysis/llm_scores_cache_{str_model_name}_{llm_model_name}.json'
    # Parse -o overrides into dict
    opt_overrides = {}
    for s in args.opt:
        s = s.strip()
        k, v = s.split('=', 1)
        keys = k.split('.')
        cur = opt_overrides
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = yaml.load(v, Loader=yaml.Loader)
    args.eval_root = str(Path(args.eval_root).expanduser().resolve())
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    entries = load_entries(args.csv)
    for e in entries:
        e['category'] = categorize(e)
    print(f'Loaded {len(entries)} samples')
    for cat in ['PL=pred', 'PL=gt', 'PL=other', 'other']:
        n = sum(1 for e in entries if e['category'] == cat)
        if n > 0:
            print(f'  {cat}: {n}')

    import json
    alpha = args.alpha
    lo_clip = args.lo_clip
    threshold = args.threshold
    use_str = not args.llm_only
    use_llm = not args.str_only
    n_entries = len(entries)

    # --- STR scores ---
    if use_str:
        str_cache_path = args.str_cache
        use_str_cache = not args.no_str_cache and os.path.exists(str_cache_path)
        per_char_list = None
        if use_str_cache:
            with open(str_cache_path, 'r') as f:
                str_cache = json.load(f)
            cached_model = str_cache.get('str_model', 'unknown')
            str_lookup = {(r['dataset_name'], r['image_index']): r for r in str_cache['scores']}
            pred_lp_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('pred_lp', 0.0) for e in entries]
            gt_lp_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('gt_lp', 0.0) for e in entries]
            print(f'Loaded STR scores from cache: {str_cache_path} (model: {cached_model})')
        else:
            pred_lp_list, gt_lp_list, per_char_list = compute_str_scores(entries, args.config, args.checkpoint, args.eval_root, args.device, opt_overrides)
            str_cache = {
                'str_model': str_model_name,
                'config': args.config,
                'checkpoint': args.checkpoint,
                'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                            'pred': e['pred'], 'gt': e['gt'],
                            'pred_lp': pred_lp_list[i], 'gt_lp': gt_lp_list[i]}
                           for i, e in enumerate(entries)],
            }
            os.makedirs(os.path.dirname(str_cache_path) or '.', exist_ok=True)
            with open(str_cache_path, 'w') as f:
                json.dump(str_cache, f, indent=2)
            print(f'STR scores cached to {str_cache_path} (model: {str_model_name})')

        pred_lp = np.array(pred_lp_list)
        gt_lp = np.array(gt_lp_list)
        lo_str = np.clip(pred_lp - gt_lp, -lo_clip, lo_clip)
        conf_str = sigmoid(lo_str)

        # Save per-char probs to Excel
        if per_char_list:
            _save_per_char_xlsx(entries, conf_str, per_char_list, args.output)
            _save_alignment_xlsx(entries, conf_str, pred_lp, gt_lp, per_char_list, args.output)

    # --- LLM scores ---
    if use_llm:
        llm_cache_path = args.llm_cache
        use_cache = not args.no_llm_cache and os.path.exists(llm_cache_path)
        if use_cache:
            with open(llm_cache_path, 'r') as f:
                cache = json.load(f)
            if isinstance(cache, list):
                scores_list = cache
                cached_model = 'unknown'
            else:
                scores_list = cache['scores']
                cached_model = cache.get('llm_model', 'unknown')
            cache_lookup = {(r['dataset_name'], r['image_index']): r.get('lo_llm', r.get('llm_score', 0.0)) for r in scores_list}
            lo_llm_list = [cache_lookup.get((e['dataset_name'], e['image_index']), 0.0) for e in entries]
            print(f'Loaded LLM scores from cache: {llm_cache_path} (model: {cached_model})')
        else:
            lo_llm_list = compute_llm_scores(entries, args.llm_model, args.device)
            cache = {'llm_model': args.llm_model,
                     'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                                 'pred': e['pred'], 'gt': e['gt'], 'lo_llm': s}
                                for e, s in zip(entries, lo_llm_list)]}
            os.makedirs(os.path.dirname(llm_cache_path) or '.', exist_ok=True)
            with open(llm_cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f'LLM scores cached to {llm_cache_path} (model: {args.llm_model})')

        lo_llm = np.clip(np.array(lo_llm_list), -lo_clip, lo_clip)
        conf_llm = sigmoid(lo_llm)

    # --- Fusion ---
    if use_str and use_llm:
        lo_fusion = alpha * lo_str + (1 - alpha) * lo_llm
        conf_fusion = sigmoid(lo_fusion)

    # --- Category-wise stats ---
    if use_str:
        print(f'\n--- Category-wise STR scores (lo_clip={lo_clip}) ---')
        print(f'{"category":<12} {"n":>4}  {"pred_lp":>10} {"gt_lp":>10} {"lo_str":>10} {"conf_str":>10}')
        print('-' * 60)
        for cat in ['PL=pred', 'PL=gt', 'PL=other', 'other']:
            mask = np.array([e['category'] == cat for e in entries])
            if not mask.any():
                continue
            print(f'{cat:<12} {mask.sum():>4}  {pred_lp[mask].mean():>10.3f} {gt_lp[mask].mean():>10.3f} {lo_str[mask].mean():>10.3f} {conf_str[mask].mean():>10.4f}')
        print(f'{"ALL":<12} {n_entries:>4}  {pred_lp.mean():>10.3f} {gt_lp.mean():>10.3f} {lo_str.mean():>10.3f} {conf_str.mean():>10.4f}')

        # Per-sample STR scores (descending by score_str)
        sorted_idx = np.argsort(-conf_str)
        has_stages = per_char_list and per_char_list[0].get('pred_ar_lp') is not None
        if has_stages:
            print(f'\n--- STR scores (descending) ---')
            print(f'{"rank":<5} {"score":>7} {"pred_lp":>8} {"gt_lp":>8} {"p_ar":>7} {"p_clz":>7} {"g_ar":>7} {"g_clz":>7} {"cat":<10} {"pred":<18} {"gt":<18} {"PL":<18} {"dataset":<14} {"idx":>5}')
            print('-' * 160)
            for rank, si in enumerate(sorted_idx):
                e = entries[si]
                pc = per_char_list[si] if per_char_list else {}
                print(f'{rank+1:<5} {conf_str[si]:>7.4f} {pred_lp[si]:>8.3f} {gt_lp[si]:>8.3f} '
                      f'{pc.get("pred_ar_lp", 0):>7.3f} {pc.get("pred_cloze_lp", 0):>7.3f} '
                      f'{pc.get("gt_ar_lp", 0):>7.3f} {pc.get("gt_cloze_lp", 0):>7.3f} '
                      f'{e["category"]:<10} {e["pred"]:<18} {e["gt"]:<18} {e["PL"]:<18} {e["dataset_name"]:<14} {e["image_index"]:>5}')
        else:
            print(f'\n--- STR scores (descending) ---')
            print(f'{"rank":<6} {"score":>8} {"pred_lp":>8} {"gt_lp":>8} {"category":<12} {"pred":<20} {"gt":<20} {"PL":<20} {"dataset":<16} {"idx":>6}')
            print('-' * 130)
            for rank, si in enumerate(sorted_idx):
                e = entries[si]
                print(f'{rank+1:<6} {conf_str[si]:>8.4f} {pred_lp[si]:>8.3f} {gt_lp[si]:>8.3f} {e["category"]:<12} {e["pred"]:<20} {e["gt"]:<20} {e["PL"]:<20} {e["dataset_name"]:<16} {e["image_index"]:>6}')

    # --- Save scores to CSV ---
    scores_path = args.output.replace('.png', '_scores.csv')
    with open(scores_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['dataset_name', 'image_index', 'pred', 'gt', 'PL', 'category']
        if use_str:
            header += ['pred_lp', 'gt_lp', 'lo_str', 'score_str']
            if has_stages:
                header += ['pred_ar_lp', 'pred_cloze_lp', 'gt_ar_lp', 'gt_cloze_lp']
        if use_llm:
            header += ['score_llm']
        if use_str and use_llm:
            header += ['score_fusion']
        w.writerow(header)
        for i, e in enumerate(entries):
            row = [e['dataset_name'], e['image_index'], e['pred'], e['gt'], e['PL'], e['category']]
            if use_str:
                row += [f'{pred_lp[i]:.4f}', f'{gt_lp[i]:.4f}', f'{lo_str[i]:.4f}', f'{conf_str[i]:.4f}']
                if has_stages:
                    pc = per_char_list[i] if per_char_list else {}
                    row += [f'{pc.get("pred_ar_lp", 0):.4f}', f'{pc.get("pred_cloze_lp", 0):.4f}',
                            f'{pc.get("gt_ar_lp", 0):.4f}', f'{pc.get("gt_cloze_lp", 0):.4f}']
            if use_llm:
                row += [f'{conf_llm[i]:.4f}']
            if use_str and use_llm:
                row += [f'{conf_fusion[i]:.4f}']
            w.writerow(row)
    print(f'Scores saved to {scores_path}')

    # --- Plot ---
    colors = {
        'PL=pred': '#e74c3c',
        'PL=gt': '#2ecc71',
        'PL=other': '#3498db',
        'other': '#95a5a6',
    }
    labels_order = ['PL=pred', 'PL=gt', 'PL=other', 'other']
    cats_present = [c for c in labels_order if any(e['category'] == c for e in entries)]
    bins = np.linspace(0.0, 1.0, 41)

    score_data = []
    if use_str:
        score_data.append(('STR only', 'σ(lo_STR)', conf_str))
    if use_llm:
        score_data.append(('LLM only', 'σ(lo_LLM)', conf_llm))
    if use_str and use_llm:
        score_data.append((f'Fusion (α={alpha})', f'α·STR + (1-α)·LLM', conf_fusion))

    n_cols = len(score_data)
    fig, axes = plt.subplots(2, max(n_cols, 1), figsize=(7 * max(n_cols, 1), 12))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (title, xlabel, scores) in enumerate(score_data):
        data_by_cat = {}
        for cat in cats_present:
            data_by_cat[cat] = scores[[i for i, e in enumerate(entries) if e['category'] == cat]]
        plot_hist(axes[0, col], data_by_cat, cats_present, colors, bins, xlabel, title)
        plot_strip(axes[1, col], data_by_cat, cats_present, colors, bins, xlabel, title)

    # --- Accuracy ---
    case12_idx = [i for i, e in enumerate(entries) if e['category'] in ('PL=pred', 'PL=gt')]
    if case12_idx:
        n = len(case12_idx)
        results = []
        if use_str:
            results.append(('STR only', conf_str))
        if use_llm:
            results.append(('LLM only', conf_llm))
        if use_str and use_llm:
            results.append(('Fusion', conf_fusion))
        print(f'\nCase1+2 accuracy (n={n}, threshold={threshold}):')
        for name, conf in results:
            correct = sum(1 for i in case12_idx
                          if (entries[i]['category'] == 'PL=pred' and conf[i] >= threshold) or
                             (entries[i]['category'] == 'PL=gt' and conf[i] < threshold))
            print(f'  {name:<14}: {correct}/{n} ({correct/n*100:.2f}%)')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
