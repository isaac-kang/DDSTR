import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--groups',
        type=str,
        default=None,
        help='comma-separated group indices to evaluate: 0=6bench, 1=u14m, 2=OST, 3=others. '
             'Default: all groups.')
    parser.add_argument(
        '--save_errors',
        action='store_true',
        default=False,
        help='Save error images to output/analysis/{dataset_name}/')
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    groups_arg = FLAGS.groups
    save_errors = FLAGS.save_errors
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    msr = False
    if 'RatioDataSet' in cfg.cfg['Eval']['dataset']['name']:
        msr = True

    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys'][
        'keep_keys'].append('real_ratio')
    trainer = Trainer(cfg, mode='eval')

    # --- Head weight cosine similarity analysis (base vs sim vs other) ---
    analyze_head_cosine(trainer)

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_root = os.path.expanduser(
        os.environ.get('EVAL_DATA_ROOT', '~/data/STR/openocr'))
    data_dirs_list = [
        [
            f'{data_root}/test/IIIT5k/', f'{data_root}/test/SVT/',
            f'{data_root}/test/IC13_857/', f'{data_root}/test/IC15_1811/',
            f'{data_root}/test/SVTP/', f'{data_root}/test/CUTE80/'
        ],
        [
            f'{data_root}/u14m/curve/', f'{data_root}/u14m/multi_oriented/',
            f'{data_root}/u14m/artistic/', f'{data_root}/u14m/contextless/',
            f'{data_root}/u14m/salient/', f'{data_root}/u14m/multi_words/',
            f'{data_root}/u14m/general/'
        ],
        [f'{data_root}/OST/weak/', f'{data_root}/OST/heavy/'],
        [
            f'{data_root}/wordart_test/', f'{data_root}/test/IC13_1015/',
            f'{data_root}/test/IC15_2077/'
        ]
    ]
    if groups_arg is not None:
        selected = [int(g) for g in groups_arg.split(',')]
        data_dirs_list = [data_dirs_list[i] for i in selected]

    cfg = cfg.cfg
    file_csv = open(
        cfg['Global']['output_dir'] + '/' +
        cfg['Global']['output_dir'].split('/')[-1] +
        '_eval_all_length_ratio.csv', 'w')
    csv_w = csv.writer(file_csv)
    cfg['Eval']['dataset']['name'] = cfg['Eval']['dataset']['name'] + 'Test'
    for data_dirs in data_dirs_list:

        acc_each = []
        acc_each_real = []
        acc_each_lower = []
        acc_each_ingore_space = []
        acc_each_ingore_space_lower = []
        acc_each_ignore_space_symbol = []
        acc_each_lower_ignore_space_symbol = []
        acc_each_num = []
        acc_each_dis = []
        each_len = {}
        each_ratio = {}
        for datadir in data_dirs:
            config_each = cfg.copy()
            if msr:
                config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            else:
                config_each['Eval']['dataset']['data_dir'] = datadir
            valid_dataloader = build_dataloader(config_each, 'Eval',
                                                trainer.logger)
            trainer.logger.info(
                f'{datadir} valid dataloader has {len(valid_dataloader)} iters'
            )
            trainer.valid_dataloader = valid_dataloader
            ds_name = datadir.rstrip('/').split('/')[-1]
            error_save_dir = None
            if save_errors:
                error_save_dir = os.path.join(
                    cfg['Global']['output_dir'], 'analysis', ds_name)
                os.makedirs(error_save_dir, exist_ok=True)
            metric = trainer.eval(
                error_save_dir=error_save_dir, dataset_name=ds_name)
            acc_each.append(metric['acc'] * 100)
            acc_each_real.append(metric['acc_real'] * 100)
            acc_each_lower.append(metric['acc_lower'] * 100)
            acc_each_ingore_space.append(metric['acc_ignore_space'] * 100)
            acc_each_ingore_space_lower.append(
                metric['acc_ignore_space_lower'] * 100)
            acc_each_ignore_space_symbol.append(
                metric['acc_ignore_space_symbol'] * 100)
            acc_each_lower_ignore_space_symbol.append(
                metric['acc_ignore_space_lower_symbol'] * 100)
            acc_each_dis.append(metric['norm_edit_dis'])
            acc_each_num.append(metric['num_samples'])

            trainer.logger.info('metric eval ***************')
            csv_w.writerow([datadir])
            for k, v in metric.items():
                trainer.logger.info('{}:{}'.format(k, v))
                if 'each' in k:
                    csv_w.writerow([k] + v)
                    if 'each_len' in k:
                        each_len[k] = each_len.get(k, []) + [np.array(v)]
                    if 'each_ratio' in k:
                        each_ratio[k] = each_ratio.get(k, []) + [np.array(v)]
        data_name = [
            data_n[:-1].split('/')[-1]
            if data_n[-1] == '/' else data_n.split('/')[-1]
            for data_n in data_dirs
        ]
        csv_w.writerow(['-'] + data_name + ['arithmetic_avg'] +
                       ['weighted_avg'])
        csv_w.writerow([''] + acc_each_num)
        avg1 = np.array(acc_each) * np.array(acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc'] + acc_each + [sum(acc_each) / len(acc_each)] +
                       [avg1.sum().tolist()])
        print(acc_each + [sum(acc_each) / len(acc_each)] +
              [avg1.sum().tolist()])
        avg1 = np.array(acc_each_dis) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['norm_edit_dis'] + acc_each_dis +
                       [sum(acc_each_dis) / len(acc_each)] +
                       [avg1.sum().tolist()])

        avg1 = np.array(acc_each_real) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['acc_real'] + acc_each_real +
                       [sum(acc_each_real) / len(acc_each_real)] +
                       [avg1.sum().tolist()])
        avg1 = np.array(acc_each_lower) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(['acc_lower'] + acc_each_lower +
                       [sum(acc_each_lower) / len(acc_each_lower)] +
                       [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ingore_space) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(
            ['acc_ignore_space'] + acc_each_ingore_space +
            [sum(acc_each_ingore_space) / len(acc_each_ingore_space)] +
            [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ingore_space_lower) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_lower'] +
                       acc_each_ingore_space_lower + [
                           sum(acc_each_ingore_space_lower) /
                           len(acc_each_ingore_space_lower)
                       ] + [avg1.sum().tolist()])
        avg1 = np.array(acc_each_ignore_space_symbol) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_symbol'] +
                       acc_each_ignore_space_symbol + [
                           sum(acc_each_ignore_space_symbol) /
                           len(acc_each_ignore_space_symbol)
                       ] + [avg1.sum().tolist()])
        avg1 = np.array(acc_each_lower_ignore_space_symbol) * np.array(
            acc_each_num) / sum(acc_each_num)
        csv_w.writerow(['acc_ignore_space_lower_symbol'] +
                       acc_each_lower_ignore_space_symbol + [
                           sum(acc_each_lower_ignore_space_symbol) /
                           len(acc_each_lower_ignore_space_symbol)
                       ] + [avg1.sum().tolist()])

        sum_all = np.array(each_len['each_len_num']).sum(0)
        for k, v in each_len.items():
            if k != 'each_len_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_len['each_len_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
            else:
                csv_w.writerow([k] + sum_all.tolist())

        sum_all = np.array(each_ratio['each_ratio_num']).sum(0)
        for k, v in each_ratio.items():
            if k != 'each_ratio_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_ratio['each_ratio_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
            else:
                csv_w.writerow([k] + sum_all.tolist())

    file_csv.close()


def _compute_and_print_cosine(weight, char_to_idx, mapping, title):
    """Compute and print cosine similarity analysis for a given weight matrix."""
    weight_norm = F.normalize(weight, dim=1)  # (num_classes, dim)
    num_classes = weight_norm.shape[0]

    results = []
    for key, entry in mapping.items():
        base_char = entry['base_char']
        sim_char = entry['confused_with']

        if base_char not in char_to_idx or sim_char not in char_to_idx:
            continue

        base_idx = char_to_idx[base_char]
        sim_idx = char_to_idx[sim_char]

        cos_base_sim = torch.dot(weight_norm[base_idx], weight_norm[sim_idx]).item()

        other_mask = torch.ones(num_classes, dtype=torch.bool)
        other_mask[base_idx] = False
        other_mask[sim_idx] = False
        cos_base_others = torch.mv(weight_norm[other_mask], weight_norm[base_idx])
        cos_base_other = cos_base_others.mean().item()

        margin = cos_base_sim - cos_base_other

        results.append({
            'key': key,
            'base': base_char,
            'sim': sim_char,
            'cos_base_sim': cos_base_sim,
            'cos_base_other': cos_base_other,
            'margin': margin,
        })

    print('=' * 80)
    print(title)
    print(f'{"key":<10} {"base":<6} {"sim":<6} {"cos(b,s)":<12} {"cos(b,o)":<12} {"margin":<12}')
    print('-' * 80)
    margin_list, cos_sim_list, cos_other_list = [], [], []
    for r in results:
        print(
            f'{r["key"]:<10} {r["base"]:<6} {r["sim"]:<6} '
            f'{r["cos_base_sim"]:<12.4f} {r["cos_base_other"]:<12.4f} {r["margin"]:<12.4f}')
        margin_list.append(r['margin'])
        cos_sim_list.append(r['cos_base_sim'])
        cos_other_list.append(r['cos_base_other'])

    print('-' * 80)
    print(
        f'{"MEAN":<10} {"":6} {"":6} '
        f'{np.mean(cos_sim_list):<12.4f} {np.mean(cos_other_list):<12.4f} {np.mean(margin_list):<12.4f}')
    print('=' * 80)


def analyze_head_cosine(trainer):
    """Analyze cosine similarity between base, sim(confused_with), and other classes."""
    decoder = trainer.model.decoder
    char_to_idx = trainer.post_process_class.dict

    mapping_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'unicode_mapping_hcp.json')
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # 1. Head weight (tgt_word_prj / head / fc / fc2)
    head_weight = None
    for attr in ['head', 'fc', 'fc2', 'tgt_word_prj']:
        layer = getattr(decoder, attr, None)
        if layer is not None and hasattr(layer, 'weight'):
            head_weight = layer.weight.data
            print(f'[head] decoder.{attr}: {head_weight.shape}')
            break
    if head_weight is not None:
        _compute_and_print_cosine(
            head_weight, char_to_idx, mapping,
            'Head Weight Cosine Similarity (base vs sim vs other)')
    else:
        print('WARNING: Could not find head weight in decoder')

    # 2. Char embedding weight (embedding.embedding)
    char_emb_weight = None
    emb = getattr(decoder, 'embedding', None)
    if emb is not None:
        inner = getattr(emb, 'embedding', None)
        if inner is not None and hasattr(inner, 'weight'):
            char_emb_weight = inner.weight.data
            print(f'[char_emb] decoder.embedding.embedding: {char_emb_weight.shape}')
    if char_emb_weight is not None:
        _compute_and_print_cosine(
            char_emb_weight, char_to_idx, mapping,
            'Char Embedding Cosine Similarity (base vs sim vs other)')
    else:
        print('WARNING: Could not find char embedding weight in decoder')


if __name__ == '__main__':
    main()
