"""Compute STR/LLM scores for PL error samples and plot distributions.

Reads PL_oracle.csv, loads images from eval LMDBs,
runs STR model for conf_str and LLM for conf_llm, then plots distributions.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/analysis/plot_score_distribution.py
"""

import argparse
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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine.config import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from utils.ckpt import load_ckpt


def load_csv(csv_path):
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'dataset_name': row['dataset_name'],
                'image_index': int(row['image_index']),
                'pred': row['pred'],
                'gt': row['gt'],
                'PL': row['PL'] if row['PL'] else '',
            })
    return entries


def categorize(entry):
    pl = entry['PL']
    if pl == '' or pl is None:
        return 'blank'
    if pl == entry['pred']:
        return 'PL=pred'
    if pl == entry['gt']:
        return 'PL=gt'
    return 'PL=other'


def load_image_from_lmdb(eval_root, dataset_name, image_index):
    """Load raw image bytes from eval LMDB."""
    lmdb_dir = os.path.join(eval_root, dataset_name)
    env = lmdb_lib.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        img_key = f'image-{image_index:09d}'.encode()
        imgbuf = txn.get(img_key)
        imgbuf = bytes(imgbuf)
    env.close()
    return imgbuf


# ===================== STR Scoring =====================

def compute_str_scores(entries, config_path, checkpoint_path, eval_root, device, opt_overrides=None):
    """Compute conf_str = exp(mean_lp(pred)) for each entry.

    Uses the final inference logits — scores pred tokens from the output distribution.
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
    if checkpoint_path:
        cfg['Global']['pretrained_model'] = checkpoint_path
    load_ckpt(model, cfg)
    model.eval().to(device)

    model_type = detect_model_type(cfg)
    scorer = get_scorer(model_type, post_process, device)

    # Build image transforms
    eval_cfg = cfg.get('Eval', cfg.get('Train', {}))
    transforms_cfg = eval_cfg.get('dataset', {}).get('transforms', [])
    img_transforms = []
    for t in transforms_cfg:
        if isinstance(t, dict):
            name = list(t.keys())[0]
            if 'Encode' not in name and 'KeepKeys' not in name:
                img_transforms.append(t)
    img_transforms.append({'RecTVResize': {'image_shape': [32, 128], 'padding': False}})
    img_transforms.append({'KeepKeys': {'keep_keys': ['image']}})
    ops = create_operators(img_transforms, cfg['Global'])

    pred_score_list = []
    print(f'Computing STR scores ({len(entries)} samples)...')
    with torch.inference_mode():
        for i, entry in enumerate(entries):
            imgbuf = load_image_from_lmdb(eval_root, entry['dataset_name'], entry['image_index'])
            data = {'image': imgbuf, 'label': entry['gt']}
            data = transform(data, ops)
            if data is None:
                pred_score_list.append(0.5)
                continue
            img = data[0]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.unsqueeze(0).to(device)

            preds_text, pred_scores, gt_scores = scorer.score(model, img, [entry['gt']])
            pred_lp = pred_scores[0].item()
            pred_score_list.append(np.exp(pred_lp))

            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(entries)}')

    del model
    torch.cuda.empty_cache()
    return pred_score_list


# ===================== LLM Scoring =====================

def build_prompt(pred, gt):
    return f"Which is more likely the true text, allowing for OCR mistakes (e.g., similar-looking characters)? 1) {pred} 2) {gt} Answer only 1 or 2."


def extract_llm_score(output):
    """Extract LLM score = exp(lp("1")) from vLLM output.

    Returns probability that pred is correct (i.e. answer is "1").
    """
    logprobs_list = output.outputs[0].logprobs
    if not logprobs_list:
        return 0.5

    generated_text = output.outputs[0].text
    char_pos = None
    for i, ch in enumerate(generated_text):
        if ch in ("1", "2"):
            char_pos = i
            break
    if char_pos is None:
        return 0.5

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
        return 0.5

    target_logprobs = logprobs_list[answer_token_idx]
    lp1 = None
    for token_id, logprob_obj in target_logprobs.items():
        decoded = logprob_obj.decoded_token
        if "1" in decoded and lp1 is None:
            lp1 = logprob_obj.logprob
            break

    if lp1 is None:
        return 0.0
    return np.exp(lp1)


def compute_llm_scores(entries, llm_model_name, device):
    """Compute LLM score = exp(lp("1")) for each entry using vLLM."""
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

    r_llm_list = [extract_llm_score(output) for output in outputs]

    del llm
    torch.cuda.empty_cache()
    return r_llm_list


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='tools/analysis/PL_oracle.csv')
    parser.add_argument('--config', '-c',
                        default='configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml',
                        help='STR model config')
    parser.add_argument('--checkpoint',
                        default='pretrained/mdiff4str_base/best.pth',
                        help='STR model checkpoint')
    parser.add_argument('--eval_root', default='~/data/STR/openocr/evaluation',
                        help='Root dir for eval LMDBs')
    parser.add_argument('--llm_model', default='Qwen/Qwen3-32B-AWQ')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Blending weight: alpha*STR + (1-alpha)*LLM')
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
    parser.add_argument('--output', default='output/analysis/score_distribution.png')
    parser.add_argument('-o', '--opt', nargs='*', default=[],
                        help='Override config options, e.g. -o Architecture.Decoder.decoding_mode=greedy')
    args = parser.parse_args()
    if args.str_cache is None:
        str_model_name = f'{Path(args.config).parent.name}-{Path(args.config).stem}'
        args.str_cache = f'tools/analysis/str_scores_cache_{str_model_name}.json'
    if args.llm_cache is None:
        llm_model_name = args.llm_model.replace('/', '-')
        args.llm_cache = f'tools/analysis/llm_scores_cache_{llm_model_name}.json'
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

    entries = load_csv(args.csv)
    for e in entries:
        e['category'] = categorize(e)
    print(f'Loaded {len(entries)} samples')
    for cat in ['PL=pred', 'PL=gt', 'PL=other', 'blank']:
        n = sum(1 for e in entries if e['category'] == cat)
        if n > 0:
            print(f'  {cat}: {n}')

    import json
    alpha = args.alpha
    threshold = args.threshold
    use_str = not args.llm_only
    use_llm = not args.str_only
    n_entries = len(entries)

    # --- STR scores ---
    if use_str:
        str_cache_path = args.str_cache
        use_str_cache = not args.no_str_cache and os.path.exists(str_cache_path)
        if use_str_cache:
            with open(str_cache_path, 'r') as f:
                str_cache = json.load(f)
            cached_model = str_cache.get('str_model', 'unknown')
            str_lookup = {(r['dataset_name'], r['image_index']): r for r in str_cache['scores']}
            pred_score_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('pred_score', 0.5) for e in entries]
            print(f'Loaded STR scores from cache: {str_cache_path} (model: {cached_model})')
        else:
            pred_score_list = compute_str_scores(entries, args.config, args.checkpoint, args.eval_root, args.device, opt_overrides)
            str_model_name = f'{Path(args.config).parent.name}-{Path(args.config).stem}'
            str_cache = {
                'str_model': str_model_name,
                'config': args.config,
                'checkpoint': args.checkpoint,
                'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                            'pred': e['pred'], 'gt': e['gt'],
                            'pred_score': pred_score_list[i]}
                           for i, e in enumerate(entries)],
            }
            os.makedirs(os.path.dirname(str_cache_path) or '.', exist_ok=True)
            with open(str_cache_path, 'w') as f:
                json.dump(str_cache, f, indent=2)
            print(f'STR scores cached to {str_cache_path} (model: {str_model_name})')

        conf_str = np.array(pred_score_list)

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
            cache_lookup = {(r['dataset_name'], r['image_index']): r.get('llm_score', r.get('r_llm', 0.5)) for r in scores_list}
            llm_score_list = [cache_lookup.get((e['dataset_name'], e['image_index']), 0.5) for e in entries]
            print(f'Loaded LLM scores from cache: {llm_cache_path} (model: {cached_model})')
        else:
            llm_score_list = compute_llm_scores(entries, args.llm_model, args.device)
            cache = {'llm_model': args.llm_model,
                     'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                                 'pred': e['pred'], 'gt': e['gt'], 'llm_score': s}
                                for e, s in zip(entries, llm_score_list)]}
            os.makedirs(os.path.dirname(llm_cache_path) or '.', exist_ok=True)
            with open(llm_cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f'LLM scores cached to {llm_cache_path} (model: {args.llm_model})')

        conf_llm = np.array(llm_score_list)

    # --- Fusion ---
    if use_str and use_llm:
        conf_fusion = alpha * conf_str + (1 - alpha) * conf_llm

    # --- Category-wise stats ---
    if use_str:
        print(f'\n--- Category-wise STR score: exp(mean_lp(pred)) ---')
        print(f'{"category":<12} {"n":>4}  {"conf_str":>12}')
        print('-' * 32)
        for cat in ['PL=pred', 'PL=gt', 'PL=other', 'blank']:
            mask = np.array([e['category'] == cat for e in entries])
            if not mask.any():
                continue
            print(f'{cat:<12} {mask.sum():>4}  {conf_str[mask].mean():>12.4f}')
        print(f'{"ALL":<12} {n_entries:>4}  {conf_str.mean():>12.4f}')

    # --- Save scores to CSV ---
    scores_path = args.output.replace('.png', '_scores.csv')
    with open(scores_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['dataset_name', 'image_index', 'pred', 'gt', 'PL', 'category']
        if use_str:
            header += ['conf_str']
        if use_llm:
            header += ['conf_llm']
        if use_str and use_llm:
            header += ['conf_fusion']
        w.writerow(header)
        for i, e in enumerate(entries):
            row = [e['dataset_name'], e['image_index'], e['pred'], e['gt'], e['PL'], e['category']]
            if use_str:
                row += [f'{conf_str[i]:.4f}']
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
        'blank': '#95a5a6',
    }
    labels_order = ['PL=pred', 'PL=gt', 'PL=other', 'blank']
    cats_present = [c for c in labels_order if any(e['category'] == c for e in entries)]
    bins = np.linspace(0.0, 1.0, 41)

    score_data = []
    if use_str:
        score_data.append(('STR only', 'exp(mean_lp(pred))', conf_str))
    if use_llm:
        score_data.append(('LLM only', 'exp(lp("1"))', conf_llm))
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
