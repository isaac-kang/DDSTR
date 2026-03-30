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


class PARSeqARScorer:
    """S(seq) via 2-stage PARSeq: AR decode → Cloze refinement scoring.

    Stage 1 (AR): greedy decode to get Y_ar_pred.
    Stage 2 (Cloze): feed Y_ar_pred into cloze refinement, score GT/pred
                     from the refinement logits.
    """

    def __init__(self, post_process, device):
        self.device = device
        self.post_process = post_process
        self.eos_id = 0
        self.bos_id = len(post_process.character) - 2
        self.pad_id = len(post_process.character) - 1

    def _encode_sequence(self, text):
        indices = []
        for ch in text:
            if ch in self.post_process.dict:
                indices.append(self.post_process.dict[ch])
        return indices

    def score(self, model, images, labels):
        """Returns (preds_text, pred_scores, gt_scores)."""
        bs = images.shape[0]
        device = images.device

        # Get encoder output
        x = images
        if hasattr(model, 'transform') and model.transform is not None:
            x = model.transform(x)
        if hasattr(model, 'encoder') and model.encoder is not None:
            x = model.encoder(x)

        decoder = model.decoder
        num_steps = decoder.max_label_length + 1
        pos_queries = decoder.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Stage 1: AR greedy decode
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), float('-inf'), device=device), 1)
        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=device)
        tgt_in[:, 0] = self.bos_id

        ar_logits = []
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
                ar_logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze(-2).argmax(-1)

            ar_logits = torch.cat(ar_logits, dim=1)  # (B, num_steps, C)
            ar_pred = ar_logits.argmax(-1)  # (B, num_steps)

        # Get pred text from AR output
        probs = ar_logits.softmax(-1)
        probs_np = probs.detach().cpu().numpy()
        pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        # Stage 2: Cloze refinement with AR pred as input
        # Build cloze mask (each position sees all other positions)
        cloze_query_mask = query_mask.clone()
        cloze_query_mask[torch.triu(
            torch.ones(num_steps, num_steps, dtype=torch.bool, device=device), 2)] = 0

        bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=device)
        cloze_in = torch.cat([bos, ar_pred[:, :-1]], dim=1)
        tgt_padding_mask = (cloze_in == self.eos_id).int().cumsum(-1) > 0

        with torch.no_grad():
            tgt_out = decoder.decode(
                cloze_in, x, tgt_mask, tgt_padding_mask,
                tgt_query=pos_queries,
                tgt_query_mask=cloze_query_mask[:, :cloze_in.shape[1]],
                pos_query=pos_queries,
            )
            refine_logits = decoder.head(tgt_out)  # (B, num_steps, C)

        log_probs = refine_logits.log_softmax(-1)  # (B, num_steps, C)

        # Score pred and GT from refinement logits
        pred_scores = torch.zeros(bs, device='cpu')
        gt_scores = torch.zeros(bs, device='cpu')

        for b in range(bs):
            # Pred score: score of AR pred tokens in refinement logits
            pred_indices = ar_pred[b].cpu().tolist()
            pred_len = 0
            pred_lp = 0.0
            for i, idx in enumerate(pred_indices):
                if idx == self.eos_id:
                    pred_lp += log_probs[b, i, idx].item()
                    pred_len += 1
                    break
                pred_lp += log_probs[b, i, idx].item()
                pred_len += 1
            pred_scores[b] = pred_lp / max(pred_len, 1)

            # GT score: score of GT tokens in refinement logits
            gt_indices = self._encode_sequence(labels[b])
            if not gt_indices:
                gt_scores[b] = float('-inf')
                continue
            gt_lp = 0.0
            gt_len = len(gt_indices) + 1  # +1 for EOS
            for i, idx in enumerate(gt_indices):
                gt_lp += log_probs[b, i, idx].item()
            gt_lp += log_probs[b, len(gt_indices), self.eos_id].item()
            gt_scores[b] = gt_lp / gt_len

        return preds_text, pred_scores, gt_scores


class MDiffScorer:
    """Score GT/pred from final inference logits of MDiff4STR.

    Runs normal inference (e.g. semi-AR decoding), takes the final output probs,
    and scores both pred and GT tokens from the same logit distribution.
    """

    def __init__(self, post_process, device):
        self.device = device
        self.post_process = post_process
        self.eos_id = 0

    def _encode_sequence(self, text):
        indices = []
        for ch in text:
            if ch in self.post_process.dict:
                indices.append(self.post_process.dict[ch])
        return indices

    def score(self, model, images, labels):
        """Returns (preds_text, pred_scores, gt_scores)."""
        with torch.no_grad():
            probs = model(images)  # (B, L+1, C)

        B, L1, C = probs.shape
        device = probs.device

        probs_np = probs.detach().cpu().numpy()
        pred_results = self.post_process(probs_np)
        preds_text = [r[0] for r in pred_results]

        pred_token_ids = probs.argmax(-1)  # (B, L+1)
        log_probs = probs.clamp(min=1e-10).log()  # (B, L+1, C)

        pred_scores = torch.zeros(B)
        gt_scores = torch.zeros(B)

        for b in range(B):
            # Pred score
            eos_hits = (pred_token_ids[b] == self.eos_id).nonzero(as_tuple=True)[0]
            eos_pos = eos_hits[0].item() if len(eos_hits) > 0 else L1 - 1
            valid_len = eos_pos + 1

            pred_tok = pred_token_ids[b, :valid_len]
            pred_log_p = log_probs[b, :valid_len].gather(1, pred_tok.unsqueeze(1)).squeeze(1)
            pred_scores[b] = pred_log_p.mean()

            # GT score from same logits
            gt_indices = self._encode_sequence(labels[b])
            if not gt_indices:
                gt_scores[b] = float('-inf')
                continue

            gt_len = len(gt_indices) + 1  # +1 for EOS
            score_len = min(L1, gt_len)
            gt_tok = torch.zeros(score_len, dtype=torch.long, device=device)
            n_chars = min(len(gt_indices), score_len)
            gt_tok[:n_chars] = torch.tensor(gt_indices[:n_chars], dtype=torch.long, device=device)
            if score_len == gt_len:
                gt_tok[len(gt_indices)] = self.eos_id

            gt_log_p = log_probs[b, :score_len].gather(1, gt_tok.unsqueeze(1)).squeeze(1)
            gt_scores[b] = gt_log_p.mean()

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
        return PARSeqARScorer(post_process, device)
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
