#!/usr/bin/env python3
"""B Step 2: LLM judge for ambiguous errors.

Reads error_info.tsv, prompts an LLM to choose between pred and gt,
extracts log-odds r_LLM = lp("1") - lp("2"), and performs logit fusion
with r_STR = pred_mean_lp - gt_mean_lp.

Outputs a TSV with columns: pred, gt, answer, r_llm, r_str, r_final
where answer is the fused decision ("1"=pred, "2"=gt).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODELS = [
    "Qwen/Qwen3-0.6B",               # 0
    "Qwen/Qwen3-1.7B",               # 1
    "Qwen/Qwen3-4B",                  # 2
    "Qwen/Qwen3-4B-Instruct-2507",   # 3
    "Qwen/Qwen3-8B",                  # 4
    "Qwen/Qwen3-14B",                 # 5
    "Qwen/Qwen3-30B-A3B",             # 6
    "meta-llama/Llama-3.2-1B-Instruct",  # 7
    "meta-llama/Llama-3.2-3B-Instruct",  # 8
    "meta-llama/Llama-3.1-8B-Instruct",  # 9
    "google/gemma-3-1b-it",           # 10
    "google/gemma-3-4b-it",           # 11
    "google/gemma-3-12b-it",          # 12
]


def load_error_info(tsv_path):
    """Load error_info.tsv entries."""
    entries = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entries.append({
                'dataset': row['dataset'],
                'sample_idx': int(row['sample_idx']),
                'pred': row['pred'],
                'gt': row['gt'],
                'pred_mean_lp': float(row['pred_mean_lp']),
                'gt_mean_lp': float(row['gt_mean_lp']),
            })
    return entries


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

    # Map char position -> token index
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


def find_best_alpha(entries, all_r_llm, pl_labels=None):
    """Sweep alpha for logit fusion: r_final = alpha * r_STR + (1-alpha) * r_LLM.
    If pl_labels provided, evaluate accuracy. Otherwise just return alpha=0.5."""
    if pl_labels is None:
        return 0.5

    eval_data = []
    for i, entry in enumerate(entries):
        pl = pl_labels.get((entry['dataset'], entry['sample_idx']))
        if pl == entry['pred']:
            gt_label = "1"
        elif pl == entry['gt']:
            gt_label = "2"
        else:
            continue
        r_str = entry['pred_mean_lp'] - entry['gt_mean_lp']
        eval_data.append((r_str, all_r_llm[i], gt_label))

    if not eval_data:
        return 0.5

    best_acc, best_alpha = 0, 0.5
    for alpha in np.arange(0.0, 1.05, 0.1):
        correct = sum(1 for r_s, r_l, gt_label in eval_data
                     if ("1" if alpha * r_s + (1 - alpha) * r_l >= 0 else "2") == gt_label)
        acc = correct / len(eval_data)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    print(f"Best alpha={best_alpha:.1f}, accuracy={best_acc*100:.2f}% (n={len(eval_data)})")
    return best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('error_info', help='error_info.tsv from extract_error_info.py')
    parser.add_argument('model_idx', type=int, help='Model index')
    parser.add_argument('--batch_size', type=int, default=0, help='0 = all at once')
    parser.add_argument('--output', default=None, help='Output TSV (default: auto-named)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Fixed alpha for fusion (default: auto-sweep if --pl_labels given, else 0.5)')
    parser.add_argument('--pl_labels', default=None,
                        help='Optional TSV with ground truth PL labels for alpha tuning')
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_name = MODELS[args.model_idx]
    model_short = model_name.split("/")[-1]
    print(f"Loading model: {model_name}")
    t0 = time.time()
    llm = LLM(model=model_name, max_model_len=512, enforce_eager=True)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)
    entries = load_error_info(args.error_info)
    print(f"Loaded {len(entries)} error entries")

    # Build prompts
    conversations = []
    chat_kwargs = {}
    if "Qwen3" in model_name:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    for entry in entries:
        prompt = build_prompt(entry['pred'], entry['gt'])
        conversations.append([{"role": "user", "content": prompt}])

    # Run inference
    outputs = llm.chat(conversations, sampling_params=sampling_params, **chat_kwargs)

    all_r_llm = []
    all_tok_answers = []
    for output in outputs:
        r_llm = extract_llm_logodds(output)
        all_r_llm.append(r_llm)
        text = output.outputs[0].text.strip()
        tok_answer = "1"
        for ch in text:
            if ch in ("1", "2"):
                tok_answer = ch
                break
        all_tok_answers.append(tok_answer)

    # Load PL labels for alpha tuning if provided
    pl_labels = None
    if args.pl_labels:
        pl_labels = {}
        with open(args.pl_labels) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                pl_labels[(row['dataset'], int(row['sample_idx']))] = row['pl']

    # Determine alpha
    if args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = find_best_alpha(entries, all_r_llm, pl_labels)

    # Write output
    output_path = args.output or f"{Path(args.error_info).stem}_{model_short}.tsv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['dataset', 'sample_idx', 'pred', 'gt', 'r_str', 'r_llm', 'r_final', 'answer'])
        for i, entry in enumerate(entries):
            r_str = entry['pred_mean_lp'] - entry['gt_mean_lp']
            r_final = alpha * r_str + (1 - alpha) * all_r_llm[i]
            answer = "1" if r_final >= 0 else "2"
            writer.writerow([
                entry['dataset'], entry['sample_idx'], entry['pred'], entry['gt'],
                f"{r_str:.4f}", f"{all_r_llm[i]:.4f}", f"{r_final:.4f}", answer,
            ])
    print(f"Saved to {output_path} (alpha={alpha:.1f})")


if __name__ == "__main__":
    main()
