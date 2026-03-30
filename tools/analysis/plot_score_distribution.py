"""Compute STR/LLM scores for PL error samples and plot distributions.

Reads error_details_PL.csv, loads images from eval LMDBs,
runs STR model for r_str and LLM for r_llm, then plots 3 distributions.

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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


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
    """Compute r_str = pred_mean_lp - gt_mean_lp for each entry using MDiff4STR."""
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

    r_str_list = []
    pred_lp_list = []
    gt_lp_list = []
    print(f'Computing STR scores ({len(entries)} samples)...')
    with torch.inference_mode():
        for i, entry in enumerate(entries):
            imgbuf = load_image_from_lmdb(eval_root, entry['dataset_name'], entry['image_index'])
            data = {'image': imgbuf, 'label': entry['gt']}
            data = transform(data, ops)
            if data is None:
                r_str_list.append(0.0)
                pred_lp_list.append(0.0)
                gt_lp_list.append(0.0)
                continue
            img = data[0]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.unsqueeze(0).to(device)

            preds_text, pred_scores, gt_scores = scorer.score(model, img, [entry['gt']])
            pred_lp = pred_scores[0].item()
            gt_lp = gt_scores[0].item()
            r_str = pred_lp - gt_lp
            r_str_list.append(r_str)
            pred_lp_list.append(pred_lp)
            gt_lp_list.append(gt_lp)

            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(entries)}')

    del model
    torch.cuda.empty_cache()
    return r_str_list, pred_lp_list, gt_lp_list


# ===================== LLM Scoring =====================

def build_prompt(pred, gt):
    return f"Which is more likely the true text, allowing for OCR mistakes (e.g., similar-looking characters)? 1) {pred} 2) {gt} Answer only 1 or 2."


def extract_llm_logodds(output):
    """Extract log-odds r_LLM = lp("1") - lp("2") from vLLM output."""
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
    """Compute r_llm = lp("1") - lp("2") for each entry using vLLM."""
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

    r_llm_list = [extract_llm_logodds(output) for output in outputs]

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
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--str_cache', default=None,
                        help='STR scores cache file (default: tools/analysis/str_scores_cache_{model}.json)')
    parser.add_argument('--no_str_cache', action='store_true',
                        help='Ignore existing cache and recompute STR scores')
    parser.add_argument('--llm_cache', default=None,
                        help='LLM scores cache file (default: tools/analysis/llm_scores_cache_{model}.json)')
    parser.add_argument('--no_llm_cache', action='store_true',
                        help='Ignore existing cache and recompute LLM scores')
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

    # STR scores: cache to avoid re-running
    import json
    str_cache_path = args.str_cache
    use_str_cache = not args.no_str_cache and os.path.exists(str_cache_path)
    if use_str_cache:
        with open(str_cache_path, 'r') as f:
            str_cache = json.load(f)
        cached_model = str_cache.get('str_model', 'unknown')
        str_lookup = {(r['dataset_name'], r['image_index']): r for r in str_cache['scores']}
        r_str_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('r_str', 0.0) for e in entries]
        pred_lp_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('pred_lp', 0.0) for e in entries]
        gt_lp_list = [str_lookup.get((e['dataset_name'], e['image_index']), {}).get('gt_lp', 0.0) for e in entries]
        print(f'Loaded STR scores from cache: {str_cache_path} (model: {cached_model})')
    else:
        r_str_list, pred_lp_list, gt_lp_list = compute_str_scores(entries, args.config, args.checkpoint, args.eval_root, args.device, opt_overrides)
        # Derive model name from checkpoint path (e.g. "pretrained/mdiff4str_base/best.pth" -> "mdiff4str_base")
        str_model_name = f'{Path(args.config).parent.name}-{Path(args.config).stem}'
        str_cache = {
            'str_model': str_model_name,
            'config': args.config,
            'checkpoint': args.checkpoint,
            'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                        'pred': e['pred'], 'gt': e['gt'],
                        'r_str': r_str_list[i], 'pred_lp': pred_lp_list[i], 'gt_lp': gt_lp_list[i]}
                       for i, e in enumerate(entries)],
        }
        os.makedirs(os.path.dirname(str_cache_path) or '.', exist_ok=True)
        with open(str_cache_path, 'w') as f:
            json.dump(str_cache, f, indent=2)
        print(f'STR scores cached to {str_cache_path} (model: {str_model_name})')

    # LLM scores: cache to avoid re-running
    llm_cache_path = args.llm_cache
    use_cache = not args.no_llm_cache and os.path.exists(llm_cache_path)
    if use_cache:
        with open(llm_cache_path, 'r') as f:
            cache = json.load(f)
        # Support both old (list) and new ({llm_model, scores}) format
        if isinstance(cache, list):
            scores_list = cache
            cached_model = 'unknown'
        else:
            scores_list = cache['scores']
            cached_model = cache.get('llm_model', 'unknown')
        cache_lookup = {(r['dataset_name'], r['image_index']): r['r_llm'] for r in scores_list}
        r_llm_list = [cache_lookup.get((e['dataset_name'], e['image_index']), 0.0) for e in entries]
        print(f'Loaded LLM scores from cache: {llm_cache_path} (model: {cached_model})')
    else:
        r_llm_list = compute_llm_scores(entries, args.llm_model, args.device)
        # Save cache with model name
        cache = {'llm_model': args.llm_model,
                 'scores': [{'dataset_name': e['dataset_name'], 'image_index': e['image_index'],
                             'pred': e['pred'], 'gt': e['gt'], 'r_llm': r}
                            for e, r in zip(entries, r_llm_list)]}
        os.makedirs(os.path.dirname(llm_cache_path) or '.', exist_ok=True)
        with open(llm_cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f'LLM scores cached to {llm_cache_path} (model: {args.llm_model})')

    alpha = args.alpha
    r_str = np.array(r_str_list)
    r_llm = np.array(r_llm_list)
    pred_lp = np.array(pred_lp_list)
    gt_lp = np.array(gt_lp_list)

    # Category-wise STR log prob stats
    print(f'\n--- Category-wise STR mean log prob ---')
    print(f'{"category":<12} {"n":>4}  {"pred_lp mean":>12} {"pred_lp std":>12}  {"gt_lp mean":>12} {"gt_lp std":>12}  {"r_str mean":>12}')
    print('-' * 88)
    for cat in ['PL=pred', 'PL=gt', 'PL=other', 'blank']:
        mask = np.array([e['category'] == cat for e in entries])
        if not mask.any():
            continue
        pm, ps = pred_lp[mask].mean(), pred_lp[mask].std()
        gm, gs = gt_lp[mask].mean(), gt_lp[mask].std()
        rm = r_str[mask].mean()
        print(f'{cat:<12} {mask.sum():>4}  {pm:>12.3f} {ps:>12.3f}  {gm:>12.3f} {gs:>12.3f}  {rm:>12.3f}')
    # Overall
    print(f'{"ALL":<12} {len(entries):>4}  {pred_lp.mean():>12.3f} {pred_lp.std():>12.3f}  {gt_lp.mean():>12.3f} {gt_lp.std():>12.3f}  {r_str.mean():>12.3f}')

    # Fusion
    r_fusion = alpha * r_str + (1 - alpha) * r_llm

    conf_str = sigmoid(r_str)
    conf_llm = sigmoid(r_llm)
    conf_fusion = sigmoid(r_fusion)

    # Save scores to CSV
    scores_path = args.output.replace('.png', '_scores.csv')
    with open(scores_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dataset_name', 'image_index', 'pred', 'gt', 'PL', 'category',
                     'pred_lp', 'gt_lp', 'r_str', 'r_llm', 'r_fusion',
                     'conf_str', 'conf_llm', 'conf_fusion'])
        for i, e in enumerate(entries):
            w.writerow([e['dataset_name'], e['image_index'], e['pred'], e['gt'], e['PL'],
                        e['category'], f'{pred_lp[i]:.4f}', f'{gt_lp[i]:.4f}',
                        f'{r_str[i]:.4f}', f'{r_llm[i]:.4f}',
                        f'{r_fusion[i]:.4f}',
                        f'{conf_str[i]:.4f}', f'{conf_llm[i]:.4f}',
                        f'{conf_fusion[i]:.4f}'])
    print(f'Scores saved to {scores_path}')

    # Plot: 2x4 (STR, LLM, Raw Fusion, Normalized Fusion)
    colors = {
        'PL=pred': '#e74c3c',
        'PL=gt': '#2ecc71',
        'PL=other': '#3498db',
        'blank': '#95a5a6',
    }
    labels_order = ['PL=pred', 'PL=gt', 'PL=other', 'blank']
    cats_present = [c for c in labels_order if any(e['category'] == c for e in entries)]
    bins = np.linspace(0.0, 1.0, 41)

    score_data = [
        ('STR only', 'σ(r_STR)', conf_str),
        ('LLM only', 'σ(r_LLM)', conf_llm),
        (f'Fusion (α={alpha})', f'σ(α·r_STR + (1-α)·r_LLM)', conf_fusion),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for col, (title, xlabel, scores) in enumerate(score_data):
        data_by_cat = {}
        for cat in cats_present:
            data_by_cat[cat] = scores[[i for i, e in enumerate(entries) if e['category'] == cat]]
        plot_hist(axes[0, col], data_by_cat, cats_present, colors, bins, xlabel, title)
        plot_strip(axes[1, col], data_by_cat, cats_present, colors, bins, xlabel, title)

    # Print accuracy
    case12_idx = [i for i, e in enumerate(entries) if e['category'] in ('PL=pred', 'PL=gt')]
    if case12_idx:
        n = len(case12_idx)
        results = [
            ('STR only', conf_str),
            ('LLM only', conf_llm),
            ('Fusion', conf_fusion),
        ]
        print(f'\nCase1+2 accuracy (n={n}):')
        for name, conf in results:
            correct = sum(1 for i in case12_idx
                          if (entries[i]['category'] == 'PL=pred' and conf[i] >= 0.5) or
                             (entries[i]['category'] == 'PL=gt' and conf[i] < 0.5))
            print(f'  {name:<14}: {correct}/{n} ({correct/n*100:.2f}%)')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
