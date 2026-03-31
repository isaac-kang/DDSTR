#!/usr/bin/env python3
"""SLD Step 1: Extract error information from STR model predictions on training data.

Runs inference on training LMDB datasets, collects samples where pred != gt,
and computes sequence-level scores S(pred) and S(gt) for SLD.

Supports 4 model types:
  - CTC (SVTRv2):   S(seq) = -CTCLoss(logits, seq) / T
  - AR (IGTR):      S(seq) = mean log P(y_t | y_{<t}, x) via AR mode
  - AR (PARSeq):    S(seq) = mean log P(y_t | y_{<t}, x) via AR decoding
  - BLC (MDiff):    S(seq) = (1/L) Σ_i log p_{τ_i}(y_i | x, z_{τ_i}) via shared-trace replay

Output: TSV with columns (dataset, sample_idx, pred, gt, pred_mean_lp, gt_mean_lp)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import lmdb as lmdb_lib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine.config import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators, transform
from utils.ckpt import load_ckpt


# ===================== Dataset =====================

class IndexedLMDBDataset(Dataset):
    """Loads images+labels from an LMDB, applies transforms, and tracks raw indices."""

    def __init__(self, lmdb_dir, ops, post_process, max_text_length):
        self.lmdb_dir = str(lmdb_dir)
        self.ops = ops
        self.post_process = post_process
        self.max_text_length = max_text_length

        env = lmdb_lib.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            self.num_samples = int(txn.get(b'num-samples').decode())
        env.close()

        # Build filtered index list (1-based LMDB indices that pass label filtering)
        self.filtered_indices = []  # list of 1-based LMDB indices
        self.labels = []
        env = lmdb_lib.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            for idx in range(1, self.num_samples + 1):
                label_key = f'label-{idx:09d}'.encode()
                label = txn.get(label_key)
                if label is None:
                    continue
                label = label.decode('utf-8').strip()
                # Filter by charset and length
                if len(label) > self.max_text_length or len(label) == 0:
                    continue
                # Check all chars are in the charset
                valid = True
                for ch in label:
                    if ch not in self.post_process.dict and ch.lower() not in [c.lower() for c in self.post_process.dict.keys()]:
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
        # data is a list after KeepKeys: [image, ...]
        img = data[0]
        raw_idx = lmdb_idx - 1  # 0-based
        return img, label, raw_idx


def collate_fn(batch):
    """Filter out None samples and collate."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, raw_indices = zip(*batch)
    # Pad images to max size in batch
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = torch.zeros(len(imgs), imgs[0].shape[0], max_h, max_w)
    for i, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        padded[i, :, :img.shape[1], :img.shape[2]] = img
    return padded, list(labels), list(raw_indices)


# ===================== Scoring Utilities =====================

def edit_distance_align(seq_a, seq_b):
    """Needleman-Wunsch alignment. Returns list of (op, a_idx|None, b_idx|None).
    op: 'match'|'sub'|'del'|'ins'
    """
    la, lb = len(seq_a), len(seq_b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    ops = []
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i - 1] == seq_b[j - 1]:
            ops.append(('match', i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('sub', i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('del', i - 1, None))
            i -= 1
        else:
            ops.append(('ins', None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def pd_gt_score(log_probs_2d, pred_chars, gt_chars, gt_token_ids, return_detail=False):
    """Score GT sequence against PD/MDM logits using edit-distance alignment.

    1. Edit-distance alignment → match1 (exact match) pairs are fixed.
    2. Between consecutive match1 pairs, collect pred positions and gt chars as segments.
    3. For each segment, monotonic DP to find best min(m, n) pairings.

    Args:
        log_probs_2d: (L, C) log softmax output from PD/MDM
        pred_chars: list of pred char strings (len = L, EOS excluded)
        gt_chars: list of GT char strings
        gt_token_ids: list of GT token ids (same length as gt_chars)
        return_detail: if True, also return alignment detail list

    Returns:
        list of log probs for matched GT chars
        (optionally) detail: list of dicts with alignment info
    """
    alignment = edit_distance_align(pred_chars, gt_chars)

    # Collect match1 pairs and non-match ops grouped into segments between match1s
    match1_entries = []  # list of (a_idx, b_idx, lp)
    segments = []  # list of {'pred_positions': [], 'gt_items': []}
    current_seg = {'pred_positions': [], 'gt_items': []}

    for op, a_idx, b_idx in alignment:
        if op == 'match':
            # Close current segment
            segments.append(current_seg)
            current_seg = {'pred_positions': [], 'gt_items': []}
            # Record match1
            lp = log_probs_2d[a_idx, gt_token_ids[b_idx]].item()
            match1_entries.append((a_idx, b_idx, lp))
        elif op == 'sub':
            current_seg['pred_positions'].append(a_idx)
            current_seg['gt_items'].append((b_idx, gt_token_ids[b_idx]))
        elif op == 'del':
            current_seg['pred_positions'].append(a_idx)
        elif op == 'ins':
            current_seg['gt_items'].append((b_idx, gt_token_ids[b_idx]))
    # Final segment after last match1
    segments.append(current_seg)

    # Collect all log probs
    all_lps = [lp for _, _, lp in match1_entries]

    # Process each segment with monotonic DP
    all_match2_pairs = []
    all_seg_skips = []
    for seg in segments:
        pp = seg['pred_positions']
        gi = seg['gt_items']
        if not pp or not gi:
            # Nothing to match — all are skips
            for a_idx in pp:
                all_seg_skips.append(('del', a_idx, None))
            for gt_idx, _ in gi:
                all_seg_skips.append(('ins', None, gt_idx))
            continue
        seg_lps, seg_pairs = _monotonic_match_dp(
            log_probs_2d, pp, gi, return_pairs=True)
        all_lps.extend(seg_lps)
        all_match2_pairs.extend(seg_pairs)
        # Skips within this segment
        matched_pred = {p for p, _, _ in seg_pairs}
        matched_gt = {g for _, (g, _), _ in seg_pairs}
        for a_idx in pp:
            if a_idx not in matched_pred:
                all_seg_skips.append(('del', a_idx, None))
        for gt_idx, _ in gi:
            if gt_idx not in matched_gt:
                all_seg_skips.append(('ins', None, gt_idx))

    if not return_detail:
        return all_lps

    # Build detail in alignment order: segment[0], match1[0], segment[1], match1[1], ...
    detail = []
    seg_details = []
    for seg_idx, seg in enumerate(segments):
        pp = seg['pred_positions']
        gi = seg['gt_items']
        # Find match2 pairs and skips for this segment
        seg_matched_pred = set()
        seg_matched_gt = set()
        seg_d = []
        for pred_pos, (gt_idx, tok_id), lp in all_match2_pairs:
            if pred_pos in pp:
                seg_d.append({
                    'match_type': 'match2', 'op': 'matched',
                    'pred_pos': pred_pos, 'pred_char': pred_chars[pred_pos],
                    'gt_idx': gt_idx, 'gt_char': gt_chars[gt_idx],
                    'gt_lp': lp,
                })
                seg_matched_pred.add(pred_pos)
                seg_matched_gt.add(gt_idx)
        for a_idx in pp:
            if a_idx not in seg_matched_pred:
                seg_d.append({
                    'match_type': 'skip', 'op': 'del',
                    'pred_pos': a_idx, 'pred_char': pred_chars[a_idx],
                    'gt_idx': None, 'gt_char': '-',
                    'gt_lp': None,
                })
        for gt_idx, _ in gi:
            if gt_idx not in seg_matched_gt:
                seg_d.append({
                    'match_type': 'skip', 'op': 'ins',
                    'pred_pos': None, 'pred_char': '-',
                    'gt_idx': gt_idx, 'gt_char': gt_chars[gt_idx],
                    'gt_lp': None,
                })
        # Sort within segment by position
        seg_d.sort(key=lambda d: (d['pred_pos'] if d['pred_pos'] is not None else 999,
                                   d['gt_idx'] if d['gt_idx'] is not None else 999))
        seg_details.append(seg_d)

    # Interleave: seg[0], match1[0], seg[1], match1[1], ..., seg[N]
    for seg_idx in range(len(segments)):
        detail.extend(seg_details[seg_idx])
        if seg_idx < len(match1_entries):
            a_idx, b_idx, lp = match1_entries[seg_idx]
            detail.append({
                'match_type': 'match1', 'op': 'match',
                'pred_pos': a_idx, 'pred_char': pred_chars[a_idx],
                'gt_idx': b_idx, 'gt_char': gt_chars[b_idx],
                'gt_lp': lp,
            })

    return all_lps, detail


def _monotonic_match_dp(log_probs_2d, pred_positions, gt_items, return_pairs=False):
    """Find best monotonic matching of min(M, N) pairs between pred_positions and gt_items.

    All min(M, N) items from the shorter side must be matched.
    Extra items from the longer side are skipped.
    Among valid matchings, pick the one maximizing total log prob.

    pred_positions: sorted list of pred position indices (length M)
    gt_items: list of (gt_char_idx, token_id), in order (length N)

    Returns (matched_lps, matched_pairs) if return_pairs else matched_lps.
    """
    if not pred_positions or not gt_items:
        return ([], []) if return_pairs else []

    M = len(pred_positions)
    N = len(gt_items)
    K = min(M, N)  # exactly K pairs must be formed

    # dp[i][j][k] = best log prob sum using pred[:i], gt[:j], with k matches so far
    # Too much memory for large K, but sequences are short (max ~25 chars)
    NEG_INF = float('-inf')
    dp = [[[NEG_INF] * (K + 1) for _ in range(N + 1)] for _ in range(M + 1)]
    dp[0][0][0] = 0.0
    choice = [[[None] * (K + 1) for _ in range(N + 1)] for _ in range(M + 1)]

    for i in range(M + 1):
        for j in range(N + 1):
            for k in range(K + 1):
                if dp[i][j][k] == NEG_INF:
                    continue
                # Skip pred[i]
                if i < M:
                    if dp[i + 1][j][k] < dp[i][j][k]:
                        dp[i + 1][j][k] = dp[i][j][k]
                        choice[i + 1][j][k] = ('skip_pred', i, j, k)
                # Skip gt[j]
                if j < N:
                    if dp[i][j + 1][k] < dp[i][j][k]:
                        dp[i][j + 1][k] = dp[i][j][k]
                        choice[i][j + 1][k] = ('skip_gt', i, j, k)
                # Match pred[i] with gt[j]
                if i < M and j < N and k < K:
                    pos = pred_positions[i]
                    _, tok_id = gt_items[j]
                    lp = log_probs_2d[pos, tok_id].item()
                    val = dp[i][j][k] + lp
                    if dp[i + 1][j + 1][k + 1] < val:
                        dp[i + 1][j + 1][k + 1] = val
                        choice[i + 1][j + 1][k + 1] = ('match', i, j, k)

    # Find best endpoint with exactly K matches
    best_val = NEG_INF
    best_i, best_j = M, N
    for i in range(M + 1):
        for j in range(N + 1):
            if dp[i][j][K] > best_val:
                best_val = dp[i][j][K]
                best_i, best_j = i, j

    # Traceback
    matched_lps = []
    matched_pairs = []
    i, j, k = best_i, best_j, K
    while k > 0 or i > 0 or j > 0:
        c = choice[i][j][k]
        if c is None:
            break
        action = c[0]
        pi, pj, pk = c[1], c[2], c[3]
        if action == 'match':
            pos = pred_positions[pi]
            gt_item = gt_items[pj]
            _, tok_id = gt_item
            lp = log_probs_2d[pos, tok_id].item()
            matched_lps.append(lp)
            matched_pairs.append((pos, gt_item, lp))
        i, j, k = pi, pj, pk

    matched_lps.reverse()
    matched_pairs.reverse()
    return (matched_lps, matched_pairs) if return_pairs else matched_lps


# ===================== Scorers =====================

class CTCScorer:
    """S(seq) = -CTCLoss(logits, seq) / T"""

    def __init__(self, post_process, device):
        self.device = device
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
        self.post_process = post_process

    def _encode_labels(self, labels):
        """Encode text labels to index sequences for CTC loss."""
        encoded = []
        lengths = []
        for label in labels:
            indices = []
            for ch in label:
                if ch in self.post_process.dict:
                    indices.append(self.post_process.dict[ch])
                # Skip unknown chars
            encoded.append(indices)
            lengths.append(len(indices))
        max_len = max(lengths) if lengths else 0
        padded = torch.zeros(len(labels), max_len, dtype=torch.long)
        for i, indices in enumerate(encoded):
            padded[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)
        return padded, torch.tensor(lengths, dtype=torch.long)

    def score(self, model, images, labels):
        """Returns (preds_text, pred_scores, gt_scores) where scores are per-sample."""
        logits = model(images)  # (B, T, C)
        T = logits.shape[1]

        # Decode predictions
        probs = logits.softmax(-1)
        if isinstance(probs, torch.Tensor):
            probs_np = probs.detach().cpu().numpy()
        pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        # CTC loss for scoring
        log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
        input_lengths = torch.full((logits.shape[0],), T, dtype=torch.long)

        # Score GT
        gt_targets, gt_lengths = self._encode_labels(labels)
        gt_targets = gt_targets.to(self.device)
        gt_lengths = gt_lengths.to(self.device)
        gt_loss = self.ctc_loss(log_probs, gt_targets, input_lengths.to(self.device), gt_lengths)
        gt_scores = (-gt_loss / T).detach().cpu()  # Higher = more plausible

        # Score pred
        pred_targets, pred_lengths = self._encode_labels(preds_text)
        pred_targets = pred_targets.to(self.device)
        pred_lengths = pred_lengths.to(self.device)
        # Handle empty predictions
        pred_lengths = pred_lengths.clamp(min=1)
        pred_loss = self.ctc_loss(log_probs, pred_targets, input_lengths.to(self.device), pred_lengths)
        pred_scores = (-pred_loss / T).detach().cpu()

        return preds_text, pred_scores, gt_scores


class IGTRARScorer:
    """S(seq) = mean log P(y_t | y_{<t}, x) via IGTR AR mode.

    For pred: use greedy AR decoding, collect per-step log probs.
    For GT: teacher forcing — feed GT tokens as prompt at each step.
    """

    def __init__(self, post_process, device):
        self.device = device
        self.post_process = post_process
        self.eos = 0
        self.bos = len(post_process.character) - 2
        self.ignore_index = len(post_process.character) - 1

    def _encode_sequence(self, text):
        """Encode text to token indices (without BOS/EOS)."""
        indices = []
        for ch in text:
            if ch in self.post_process.dict:
                indices.append(self.post_process.dict[ch])
        return indices

    def score(self, model, images, labels):
        """Returns (preds_text, pred_scores, gt_scores)."""
        # Full forward for predictions
        with torch.no_grad():
            logits = model(images)

        # Decode predictions
        if isinstance(logits, (list, tuple)):
            pred_results = self.post_process(logits)
        else:
            probs = logits.softmax(-1)
            probs_np = probs.detach().cpu().numpy()
            pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        if isinstance(logits, (list, tuple)):
            # IGTR returns [idx, prob] in some modes
            pred_probs_tensor = logits[1]  # (B, L) probs
            pred_scores = pred_probs_tensor.log().mean(dim=-1).detach().cpu()
        else:
            # logits: (B, L, C) — get per-position max log prob
            probs = logits.softmax(-1)
            max_probs = probs.max(-1).values  # (B, L)
            # Mask after EOS
            pred_idx = logits.argmax(-1)
            eos_mask = (pred_idx == self.eos).cumsum(-1) > 0
            # First EOS should count, shift mask
            first_eos = (pred_idx == self.eos).float().argmax(-1)
            for b in range(eos_mask.shape[0]):
                if first_eos[b] < eos_mask.shape[1]:
                    eos_mask[b, :first_eos[b] + 1] = False
            max_probs = max_probs.masked_fill(eos_mask, 1.0)
            valid_len = (~eos_mask).sum(-1).clamp(min=1).float()
            pred_scores = (max_probs.log().sum(-1) / valid_len).detach().cpu()

        # GT scoring via teacher forcing
        # We need to access the decoder directly
        gt_scores = self._score_gt_teacher_forcing(model, images, labels)

        return preds_text, pred_scores, gt_scores

    def _score_gt_teacher_forcing(self, model, images, labels):
        """Score GT sequences via AR teacher forcing on the decoder."""
        bs = images.shape[0]
        device = images.device

        # Get encoder output
        x = images
        if hasattr(model, 'transform') and model.transform is not None:
            x = model.transform(x)
        if hasattr(model, 'encoder') and model.encoder is not None:
            x = model.encoder(x)

        decoder = model.decoder

        # Setup visual features (same as forward_test)
        if not decoder.ds:
            visual_f = x + decoder.vis_pos_embed
        elif decoder.pos2d:
            x = x + decoder.vis_pos_embed[:, :, :x.shape[2], :x.shape[3]]
            visual_f = x.flatten(2).transpose(1, 2)
        else:
            visual_f = x

        ques_all = torch.tile(decoder.char_pos_embed.unsqueeze(0), (bs, 1, 1))

        scores = torch.zeros(bs, device='cpu')
        for b in range(bs):
            gt_indices = self._encode_sequence(labels[b])
            if not gt_indices:
                scores[b] = float('-inf')
                continue

            # AR teacher forcing: feed GT tokens step by step
            max_len = decoder.max_len if hasattr(decoder, 'max_len') else len(gt_indices) + 1
            tgt_in = torch.full((1, max_len), self.ignore_index, dtype=torch.long, device=device)
            tgt_in[0, 0] = self.bos

            log_probs = []
            for j_idx, gt_token in enumerate(gt_indices):
                j = j_idx + 1  # position (0 is BOS)
                if j >= max_len:
                    break

                visual_f_ar = visual_f[b:b+1]
                ques_i = ques_all[b:b+1, j:j+1, :]
                prompt_ar = ques_all[b:b+1, :j] + decoder.char_embed(tgt_in[:, :j])
                mask = torch.where(
                    (tgt_in[:, :j] == self.eos).int().cumsum(-1) > 0,
                    float('-inf'), 0.0)
                for layer in decoder.cmff_decoder:
                    ques_i, prompt_ar, visual_f_ar = layer(
                        ques_i, prompt_ar, visual_f_ar, mask.unsqueeze(1))
                answer_query_i = decoder.answer_to_question_layer(
                    ques_i, prompt_ar, mask.unsqueeze(1))
                answer_pred_i = decoder.norm_pred(
                    decoder.answer_to_image_layer(answer_query_i, visual_f_ar))
                p_i = decoder.ques1_head(answer_pred_i)  # (1, 1, C)

                log_p = p_i.squeeze().log_softmax(-1)
                log_probs.append(log_p[gt_token].item())

                # Feed GT token for next step
                tgt_in[0, j] = gt_token

            # Also score EOS token after the last GT char
            j = len(gt_indices) + 1
            if j < max_len:
                visual_f_ar = visual_f[b:b+1]
                ques_i = ques_all[b:b+1, j:j+1, :]
                prompt_ar = ques_all[b:b+1, :j] + decoder.char_embed(tgt_in[:, :j])
                mask = torch.where(
                    (tgt_in[:, :j] == self.eos).int().cumsum(-1) > 0,
                    float('-inf'), 0.0)
                for layer in decoder.cmff_decoder:
                    ques_i, prompt_ar, visual_f_ar = layer(
                        ques_i, prompt_ar, visual_f_ar, mask.unsqueeze(1))
                answer_query_i = decoder.answer_to_question_layer(
                    ques_i, prompt_ar, mask.unsqueeze(1))
                answer_pred_i = decoder.norm_pred(
                    decoder.answer_to_image_layer(answer_query_i, visual_f_ar))
                p_i = decoder.ques1_head(answer_pred_i)
                log_p = p_i.squeeze().log_softmax(-1)
                log_probs.append(log_p[self.eos].item())

            if log_probs:
                scores[b] = sum(log_probs) / len(log_probs)
            else:
                scores[b] = float('-inf')

        return scores


class PARSeqScorer:
    """Score PARSeq predictions via AR + Cloze (MLM) stages.

    Pred score: AR step argmax probs + cloze step argmax probs, all collected, mean log prob.
    GT score:   AR teacher forcing probs + cloze MLM (GT input → GT output) probs, mean log prob.
    """

    def __init__(self, post_process, device):
        self.device = device
        self.post_process = post_process
        self.eos_id = 0
        self.bos_id = len(post_process.character) - 2
        self.pad_id = len(post_process.character) - 1

    def _encode_sequence(self, text):
        return [self.post_process.dict[ch] for ch in text if ch in self.post_process.dict]

    def score(self, model, images, labels, **kwargs):
        """Returns (preds_text, pred_scores, gt_scores)."""
        bs = images.shape[0]
        device = images.device

        x = images
        if hasattr(model, 'transform') and model.transform is not None:
            x = model.transform(x)
        if hasattr(model, 'encoder') and model.encoder is not None:
            x = model.encoder(x)

        decoder = model.decoder
        num_steps = decoder.max_label_length + 1
        pos_queries = decoder.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), float('-inf'), device=device), 1)

        # ===== Stage 1: AR decoding =====
        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=device)
        tgt_in[:, 0] = self.bos_id

        ar_logits_list = []
        with torch.no_grad():
            for i in range(num_steps):
                j = i + 1
                tgt_out = decoder.decode(
                    tgt_in[:, :j], x,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                    pos_query=pos_queries,
                )
                p_i = decoder.head(tgt_out)
                ar_logits_list.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze(-2).argmax(-1)

        ar_logits = torch.cat(ar_logits_list, dim=1)  # (B, num_steps, C)
        ar_probs = ar_logits.softmax(-1)
        ar_log_probs = ar_logits.log_softmax(-1)
        ar_pred_ids = ar_logits.argmax(-1)  # (B, num_steps)

        # ===== Stage 2: Cloze refinement (if refine_iters > 0) =====
        cloze_log_probs = None
        cloze_pred_ids = None
        if decoder.refine_iters:
            cloze_query_mask = query_mask.clone()
            cloze_query_mask[torch.triu(
                torch.ones(num_steps, num_steps, dtype=torch.bool, device=device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=device)
            logits = ar_logits
            with torch.no_grad():
                for _ in range(decoder.refine_iters):
                    cloze_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                    tgt_padding_mask = (cloze_in == self.eos_id).int().cumsum(-1) > 0
                    tgt_out = decoder.decode(
                        cloze_in, x, tgt_mask, tgt_padding_mask,
                        tgt_query=pos_queries,
                        tgt_query_mask=cloze_query_mask[:, :cloze_in.shape[1]],
                        pos_query=pos_queries,
                    )
                    logits = decoder.head(tgt_out)
            cloze_probs = logits.softmax(-1)
            cloze_log_probs = logits.log_softmax(-1)
            cloze_pred_ids = logits.argmax(-1)

        # Final probs for post_process (use cloze if available, else AR)
        final_probs = cloze_probs if cloze_log_probs is not None else ar_probs
        probs_np = final_probs.detach().cpu().numpy()
        pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]
        final_pred_ids = cloze_pred_ids if cloze_pred_ids is not None else ar_pred_ids

        # ===== Compute pred_score and gt_score per sample =====
        pred_scores = torch.zeros(bs, device='cpu')
        gt_scores = torch.zeros(bs, device='cpu')
        self.last_char_probs = []
        # Per-stage breakdown (for last sample in batch)
        self.last_pred_ar_lp = 0.0
        self.last_pred_cloze_lp = 0.0
        self.last_gt_ar_lp = 0.0
        self.last_gt_cloze_lp = 0.0

        for b in range(bs):
            # Determine pred length (up to first EOS in final output)
            eos_hits = (final_pred_ids[b] == self.eos_id).nonzero(as_tuple=True)[0]
            pred_len = eos_hits[0].item() if len(eos_hits) > 0 else num_steps - 1

            if pred_len == 0:
                pred_scores[b] = float('-inf')
                gt_scores[b] = float('-inf')
                continue

            # --- Pred score: collect all stage argmax probs ---
            pred_ar_lps = []
            for i in range(pred_len):
                pred_ar_lps.append(ar_log_probs[b, i, ar_pred_ids[b, i]].item())
            pred_cloze_lps = []
            if cloze_log_probs is not None:
                for i in range(pred_len):
                    pred_cloze_lps.append(cloze_log_probs[b, i, cloze_pred_ids[b, i]].item())
            pred_all = pred_ar_lps + pred_cloze_lps
            pred_scores[b] = sum(pred_all) / len(pred_all)

            # --- GT score: AR teacher forcing + Cloze MLM ---
            gt_indices = self._encode_sequence(labels[b])
            if not gt_indices:
                gt_scores[b] = float('-inf')
                continue

            gt_len = len(gt_indices)

            # AR teacher forcing
            gt_ar_lps = []
            gt_tgt_in = torch.full((1, num_steps), self.pad_id, dtype=torch.long, device=device)
            gt_tgt_in[0, 0] = self.bos_id
            with torch.no_grad():
                for i in range(min(gt_len + 1, num_steps)):
                    j = i + 1
                    tgt_out = decoder.decode(
                        gt_tgt_in[:, :j], x[b:b+1],
                        tgt_mask[:j, :j],
                        tgt_query=pos_queries[b:b+1, i:j],
                        tgt_query_mask=query_mask[i:j, :j],
                        pos_query=pos_queries[b:b+1],
                    )
                    p_i = decoder.head(tgt_out)
                    log_p = p_i.squeeze().log_softmax(-1)
                    if i < gt_len:
                        gt_ar_lps.append(log_p[gt_indices[i]].item())
                        gt_tgt_in[0, j] = gt_indices[i]
                    else:
                        gt_ar_lps.append(log_p[self.eos_id].item())

            # Cloze MLM: GT as input → GT output prob
            gt_cloze_lps = []
            if decoder.refine_iters:
                gt_cloze_in_ids = [self.bos_id] + gt_indices
                gt_cloze_in_ids = gt_cloze_in_ids + [self.pad_id] * (num_steps - len(gt_cloze_in_ids))
                gt_cloze_in = torch.tensor([gt_cloze_in_ids[:num_steps]], dtype=torch.long, device=device)
                tgt_padding_mask = (gt_cloze_in == self.eos_id).int().cumsum(-1) > 0
                cloze_qm = query_mask.clone()
                cloze_qm[torch.triu(
                    torch.ones(num_steps, num_steps, dtype=torch.bool, device=device), 2)] = 0
                with torch.no_grad():
                    tgt_out = decoder.decode(
                        gt_cloze_in, x[b:b+1], tgt_mask, tgt_padding_mask,
                        tgt_query=pos_queries[b:b+1],
                        tgt_query_mask=cloze_qm[:, :gt_cloze_in.shape[1]],
                        pos_query=pos_queries[b:b+1],
                    )
                    gt_cloze_logits = decoder.head(tgt_out)
                gt_cloze_lp_tensor = gt_cloze_logits.log_softmax(-1)
                for i in range(gt_len):
                    gt_cloze_lps.append(gt_cloze_lp_tensor[0, i, gt_indices[i]].item())
                gt_cloze_lps.append(gt_cloze_lp_tensor[0, gt_len, self.eos_id].item())

            gt_all = gt_ar_lps + gt_cloze_lps
            gt_scores[b] = sum(gt_all) / len(gt_all)

            # Save per-stage breakdown for last sample
            self.last_pred_ar_lp = sum(pred_ar_lps) / len(pred_ar_lps) if pred_ar_lps else 0.0
            self.last_pred_cloze_lp = sum(pred_cloze_lps) / len(pred_cloze_lps) if pred_cloze_lps else 0.0
            self.last_gt_ar_lp = sum(gt_ar_lps) / len(gt_ar_lps) if gt_ar_lps else 0.0
            self.last_gt_cloze_lp = sum(gt_cloze_lps) / len(gt_cloze_lps) if gt_cloze_lps else 0.0

        # Per-char probs for last sample
        b = bs - 1
        eos_hits = (final_pred_ids[b] == self.eos_id).nonzero(as_tuple=True)[0]
        eos_pos = eos_hits[0].item() if len(eos_hits) > 0 else num_steps - 1
        self.last_char_probs = final_probs[b, :eos_pos].max(-1).values.detach().cpu().tolist()

        return preds_text, pred_scores, gt_scores


class MDiffScorer:
    """Score MDiff4STR via final inference logits.

    Pred score: geometric mean of argmax probs from final logits.
    GT score:   PD-style alignment scoring (edit-distance + monotonic matching).
    """

    def __init__(self, post_process, device):
        self.device = device
        self.post_process = post_process
        self.eos_id = 0

    def _encode_sequence(self, text):
        return [self.post_process.dict[ch] for ch in text if ch in self.post_process.dict]

    def score(self, model, images, labels, **kwargs):
        """Returns (preds_text, pred_scores, gt_scores)."""
        with torch.no_grad():
            probs = model(images)  # (B, L+1, C) softmaxed

        B, L1, C = probs.shape
        probs_np = probs.detach().cpu().numpy()
        pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        pred_token_ids = probs.argmax(-1)
        max_probs = probs.max(-1).values
        log_probs = probs.clamp(min=1e-10).log()

        pred_scores = torch.zeros(B)
        gt_scores = torch.zeros(B)
        self.last_char_probs = []
        self.last_alignment_detail = None

        for b in range(B):
            eos_hits = (pred_token_ids[b] == self.eos_id).nonzero(as_tuple=True)[0]
            eos_pos = eos_hits[0].item() if len(eos_hits) > 0 else L1 - 1
            pred_len = eos_pos  # EOS 제외

            if pred_len == 0:
                pred_scores[b] = float('-inf')
                gt_scores[b] = float('-inf')
                continue

            # Pred score: argmax probs
            pred_scores[b] = max_probs[b, :pred_len].log().mean()

            # GT score: PD alignment
            pred_chars = list(preds_text[b])
            gt_indices = self._encode_sequence(labels[b])
            gt_chars = list(labels[b])

            if not gt_indices:
                gt_scores[b] = float('-inf')
                continue

            gt_lps, detail = pd_gt_score(
                log_probs[b], pred_chars, gt_chars, gt_indices, return_detail=True)
            gt_scores[b] = sum(gt_lps) / len(gt_lps) if gt_lps else float('-inf')
            self.last_alignment_detail = detail

        b = B - 1
        eos_hits = (pred_token_ids[b] == self.eos_id).nonzero(as_tuple=True)[0]
        eos_pos = eos_hits[0].item() if len(eos_hits) > 0 else L1 - 1
        self.last_char_probs = max_probs[b, :eos_pos].detach().cpu().tolist()

        return preds_text, pred_scores, gt_scores


# ===================== Model Loading =====================

def load_model(config_path, checkpoint_path, device):
    """Load an OpenOCR model from config and checkpoint."""
    cfg = Config(config_path).cfg

    # Build post process to get charset
    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    # Build model
    model = build_model(cfg['Architecture'])

    # Load weights
    if checkpoint_path:
        cfg['Global']['pretrained_model'] = checkpoint_path
    load_ckpt(model, cfg)

    model.eval().to(device)
    return model, cfg, post_process


def detect_model_type(cfg):
    """Detect model type from config."""
    decoder_name = cfg['Architecture']['Decoder']['name']
    if 'CTC' in decoder_name or 'RCTC' in decoder_name:
        return 'ctc'
    elif 'IGTR' in decoder_name:
        return 'igtr_ar'
    elif 'PARSeq' in decoder_name:
        return 'parseq_ar'
    elif 'MDiff' in decoder_name:
        return 'mdiff_blc'
    else:
        raise ValueError(f'Unknown decoder type: {decoder_name}. '
                         f'Supported: CTCDecoder, RCTCDecoder, IGTRDecoder, PARSeqDecoder, MDiffDecoder')


def get_scorer(model_type, post_process, device):
    """Get the appropriate scorer for the model type."""
    if model_type == 'ctc':
        return CTCScorer(post_process, device)
    elif model_type == 'igtr_ar':
        return IGTRARScorer(post_process, device)
    elif model_type == 'parseq_ar':
        return PARSeqScorer(post_process, device)
    elif model_type == 'mdiff_blc':
        return MDiffScorer(post_process, device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


# ===================== Main =====================

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='SLD Step 1: Extract error info')
    parser.add_argument('--config', '-c', required=True, help='Model config YAML')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint path')
    parser.add_argument('--data_root', default='~/data/STR/openocr',
                        help='Root dir containing training LMDBs')
    parser.add_argument('--train_dirs', nargs='+',
                        default=['Union14M-L-LMDB-Filtered/filter_train_challenging',
                                 'Union14M-L-LMDB-Filtered/filter_train_hard',
                                 'Union14M-L-LMDB-Filtered/filter_train_medium',
                                 'Union14M-L-LMDB-Filtered/filter_train_normal',
                                 'Union14M-L-LMDB-Filtered/filter_train_easy'],
                        help='Training LMDB directories relative to data_root')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None,
                        help='Output TSV path (default: ~/data/STR/ddstr/error_info/{model_type}/error_info.tsv)')
    parser.add_argument('--model_type', default=None,
                        help='Override model type detection (ctc, igtr_ar, parseq_ar, mdiff_blc)')
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())

    # Load model
    print(f'Loading model from {args.config}...')
    model, cfg, post_process = load_model(args.config, args.checkpoint, args.device)

    # Detect model type
    model_type = args.model_type or detect_model_type(cfg)
    print(f'Model type: {model_type}')

    # Setup scorer
    scorer = get_scorer(model_type, post_process, args.device)

    # Setup output path
    if args.output is None:
        output_dir = Path.home() / 'data' / 'STR' / 'ddstr' / 'error_info' / model_type
        output_path = output_dir / 'error_info.tsv'
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build transforms: eval-mode image loading + resize + normalize (no augmentation)
    # Use Eval transforms to avoid training augmentations (e.g. PARSeqAugPIL)
    eval_cfg = cfg.get('Eval', cfg.get('Train', {}))
    dataset_cfg = eval_cfg.get('dataset', {})
    transforms_cfg = dataset_cfg.get('transforms', [])
    img_transforms = []
    for t in transforms_cfg:
        if isinstance(t, dict):
            name = list(t.keys())[0]
            if 'Encode' not in name and 'KeepKeys' not in name:
                img_transforms.append(t)
    # RecTVResize: PIL Image -> resize to (32, 128) -> ToTensor -> Normalize(0.5, 0.5)
    img_transforms.append({'RecTVResize': {'image_shape': [32, 128], 'padding': False}})
    img_transforms.append({'KeepKeys': {'keep_keys': ['image']}})
    ops = create_operators(img_transforms, cfg['Global'])

    max_text_length = cfg['Global'].get('max_text_length', 25)

    # Process each training directory
    errors = []
    for train_dir in args.train_dirs:
        lmdb_path = Path(args.data_root) / train_dir
        if not lmdb_path.exists():
            print(f'WARNING: {lmdb_path} does not exist, skipping')
            continue

        # Check if this is a leaf LMDB or contains sub-LMDBs
        if (lmdb_path / 'data.mdb').exists():
            lmdb_dirs = [lmdb_path]
        else:
            lmdb_dirs = sorted(p.parent for p in lmdb_path.rglob('data.mdb'))

        for lmdb_dir in lmdb_dirs:
            ds_name = str(lmdb_dir.relative_to(Path(args.data_root)))
            print(f'Processing {ds_name}...')

            dataset = IndexedLMDBDataset(lmdb_dir, ops, post_process, max_text_length)
            if len(dataset) == 0:
                print(f'  Skipping {ds_name}: no valid samples')
                continue

            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                collate_fn=collate_fn, pin_memory=True)

            for batch in tqdm(dataloader, desc=ds_name):
                if batch is None:
                    continue
                imgs, labels, raw_indices = batch
                imgs = imgs.to(args.device)

                preds_text, pred_scores, gt_scores = scorer.score(model, imgs, labels)

                for i, (pred, gt) in enumerate(zip(preds_text, labels)):
                    # Case-insensitive: treat case-only difference as correct
                    if pred.lower() != gt.lower():
                        errors.append({
                            'dataset': ds_name,
                            'sample_idx': raw_indices[i],
                            'pred': pred,
                            'gt': gt,
                            'pred_mean_lp': round(pred_scores[i].item(), 6),
                            'gt_mean_lp': round(gt_scores[i].item(), 6),
                        })

    # Sort by pred_mean_lp ascending (least confident first)
    errors.sort(key=lambda e: e['pred_mean_lp'])

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['dataset', 'sample_idx', 'pred', 'gt', 'pred_mean_lp', 'gt_mean_lp'],
            delimiter='\t')
        writer.writeheader()
        writer.writerows(errors)

    print(f'Saved {len(errors)} errors to {output_path}')


if __name__ == '__main__':
    main()
