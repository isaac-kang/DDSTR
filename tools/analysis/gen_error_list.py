"""Generate error list Excel for PL labeling.

Uses the same eval pipeline as eval_rec_all_en.py (Trainer + RatioDataSetTVResizeTest).
Collects pred != gt samples and saves to Excel with columns for manual PL annotation.

Usage:
    python tools/analysis/gen_error_list.py --model mdiff4str
    python tools/analysis/gen_error_list.py --model parseq
    python tools/analysis/gen_error_list.py -c configs/rec/... -o Global.pretrained_model=pretrained/.../best.pth
"""

import argparse
import io
import os
import string
import sys
from pathlib import Path

import lmdb as lmdb_lib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer

# ===================== Model presets =====================

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

EVAL_DATASETS = ['IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80']


def _normalize(text):
    """Lowercase + keep only alphanumeric (standard STR eval)."""
    return ''.join(ch for ch in text.lower() if ch in string.digits + string.ascii_lowercase)


def load_image_from_lmdb(lmdb_dir, lmdb_index):
    """Load raw image bytes from LMDB (1-based index)."""
    env = lmdb_lib.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(buffers=True) as txn:
        img_key = f'image-{lmdb_index:09d}'.encode()
        imgbuf = txn.get(img_key)
        imgbuf = bytes(imgbuf) if imgbuf else None
    env.close()
    return imgbuf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_PRESETS.keys()),
                        help='Model preset name')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('--eval_root', default='~/data/STR/openocr/evaluation',
                        help='Root dir for eval LMDBs')
    parser.add_argument('--output', default=None,
                        help='Output Excel path (default: output/analysis/error_list_{model}.xlsx)')
    parser.add_argument('-o', '--opt', nargs='*', default=[])
    args = parser.parse_args()

    # Resolve config from preset or args
    if args.model:
        preset = MODEL_PRESETS[args.model]
        if args.config is None:
            args.config = preset['config']
    if args.config is None:
        parser.error('Either --model or --config is required')
    model_name = args.model or Path(args.config).stem

    cfg = Config(args.config)
    # Parse -o overrides
    import yaml
    opt = {}
    for s in (args.opt or []):
        s = s.strip()
        k, v = s.split('=', 1)
        keys = k.split('.')
        cur = opt
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = yaml.safe_load(v)
    cfg.merge_dict(opt)

    # Apply pretrained_model from preset if not overridden
    if args.model and cfg.cfg['Global'].get('pretrained_model') is None:
        cfg.cfg['Global']['pretrained_model'] = MODEL_PRESETS[args.model]['checkpoint']

    if args.output is None:
        args.output = f'output/analysis/error_list_{model_name}.xlsx'
    args.eval_root = str(Path(args.eval_root).expanduser().resolve())

    # Increase eval batch size for faster inference
    cfg.cfg['Eval']['loader']['batch_size_per_card'] = 1024
    cfg.cfg['Eval']['sampler']['first_bs'] = 1024

    # Setup config same as eval_rec_all_en.py
    msr = 'RatioDataSet' in cfg.cfg['Eval']['dataset']['name']
    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys'][
        'keep_keys'].append('real_ratio')
    cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys'][
        'keep_keys'].append('file_idx')

    trainer = Trainer(cfg, mode='eval')
    cfgd = cfg.cfg
    cfgd['Eval']['dataset']['name'] = cfgd['Eval']['dataset']['name'] + 'Test'

    print(f'Model: {model_name}')
    print(f'Checkpoint: {cfgd["Global"]["pretrained_model"]}')

    # Collect errors across all 6 benchmarks
    errors = []
    total = 0
    correct = 0

    # Track cumulative sample index per dataset for LMDB mapping
    for ds_name in EVAL_DATASETS:
        data_dir = os.path.join(args.eval_root, ds_name)
        if not os.path.exists(data_dir):
            print(f'  {ds_name}: not found, skipping')
            continue

        config_each = cfgd.copy()
        if msr:
            config_each['Eval']['dataset']['data_dir_list'] = [data_dir]
        else:
            config_each['Eval']['dataset']['data_dir'] = data_dir

        import logging, io as _io, contextlib
        trainer.logger.setLevel(logging.WARNING)
        with contextlib.redirect_stdout(_io.StringIO()):
            valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
        trainer.logger.setLevel(logging.INFO)

        trainer.valid_dataloader = valid_dataloader
        ds_errors = 0
        ds_total = 0

        trainer.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_dataloader, desc=ds_name)):
                batch_tensor = [t.to(trainer.device) for t in batch]
                batch_numpy = [t.numpy() for t in batch]

                preds = trainer.model(batch_tensor[0], data=batch_tensor[1:])
                post_result = trainer.post_process_class(preds, batch_numpy)

                # batch: image, label, length, real_ratio, file_idx
                file_idx_batch = batch_numpy[-1]  # file_idx is last element

                if isinstance(post_result, tuple):
                    cur_preds, cur_labels = post_result
                    for i in range(len(cur_preds)):
                        pred_text = cur_preds[i][0] if isinstance(cur_preds[i], tuple) else cur_preds[i]
                        gt_text = cur_labels[i][0] if isinstance(cur_labels[i], tuple) else cur_labels[i]
                        lmdb_idx = int(file_idx_batch[i])

                        ds_total += 1
                        total += 1

                        if _normalize(pred_text) == _normalize(gt_text):
                            correct += 1
                        else:
                            errors.append({
                                'dataset': ds_name,
                                'lmdb_index': lmdb_idx,
                                'pred': pred_text,
                                'gt': gt_text,
                            })
                            ds_errors += 1

        print(f'  {ds_name}: {ds_total} samples, {ds_errors} errors')

    print(f'\nTotal: {total}, Correct: {correct}, Errors: {len(errors)}')
    print(f'Accuracy: {correct/total*100:.2f}%')

    # Save to Excel
    _save_excel(errors, args.output, args.eval_root)


def _save_excel(errors, output_path, eval_root):
    """Save error list to Excel and images."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        import csv
        csv_path = output_path.replace('.xlsx', '.csv')
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'lmdb_index', 'pred', 'gt', 'PL_label'])
            for e in errors:
                w.writerow([e['dataset'], e['lmdb_index'], e['pred'], e['gt'], ''])
        print(f'Saved to {csv_path}')
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Error List'

    headers = ['dataset', 'lmdb_index', 'pred', 'gt', 'PL_label']
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color='DAEEF3', end_color='DAEEF3', fill_type='solid')
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = header_font
        c.fill = header_fill

    ws.cell(row=2, column=5, value='1=PL is pred, 2=PL is gt, 3=other/illegible')
    ws.cell(row=2, column=5).font = Font(italic=True, color='888888')

    yellow_fill = PatternFill(start_color='FFFFCC', end_color='FFFFCC', fill_type='solid')
    row = 3
    for e in errors:
        ws.cell(row=row, column=1, value=e['dataset'])
        ws.cell(row=row, column=2, value=e['lmdb_index'])
        ws.cell(row=row, column=3, value=e['pred'])
        ws.cell(row=row, column=4, value=e['gt'])
        ws.cell(row=row, column=5, value='').fill = yellow_fill
        row += 1

    for col in range(1, 6):
        max_len = max(len(str(ws.cell(row=r, column=col).value or '')) for r in range(1, row))
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = min(max_len + 2, 30)

    wb.save(output_path)
    print(f'Saved {len(errors)} errors to {output_path}')

    # Save error images (clear existing)
    import shutil
    img_dir = output_path.replace('.xlsx', '_images').replace('.csv', '_images')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    saved = 0
    for i, e in enumerate(errors):
        lmdb_dir = os.path.join(eval_root, e['dataset'])
        if e['lmdb_index'] <= 0:
            continue
        imgbuf = load_image_from_lmdb(lmdb_dir, e['lmdb_index'])
        if imgbuf is None:
            continue
        try:
            pil_img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
        except Exception:
            continue
        fname = f'{i:04d}_{e["dataset"]}_{e["lmdb_index"]}.png'
        pil_img.save(os.path.join(img_dir, fname))
        saved += 1
    print(f'Saved {saved} images to {img_dir}/')
    print(f'Fill in PL_label column: 1=pred, 2=gt, 3=other/illegible')


if __name__ == '__main__':
    main()
