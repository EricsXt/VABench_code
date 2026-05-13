import argparse
import csv
import json
import os
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import cls_data_generator
import cls_feature_class
import parameters
import seldnet_model
from cls_compute_seld_results import ComputeSELDResults
from train_seldnet import test_epoch


def get_dev_splits(params):
    if params.get('split_strategy') == 'manifest':
        return ['train'], ['valid'], ['test']
    if '2020' in params['dataset_dir']:
        return [[3, 4, 5, 6]], [2], [1]
    if '2021' in params['dataset_dir']:
        return [[1, 2, 3, 4]], [5], [6]
    if '2022' in params['dataset_dir']:
        return [[1, 2, 3]], [[4]], [[4]]
    if '2023' in params['dataset_dir'] or 'STARSS23' in params['dataset_dir']:
        return [[1, 2, 3]], [[4]], [[4]]
    if '2024' in params['dataset_dir']:
        return [[3]], [[4]], [[4]]
    raise ValueError('Unknown dataset splits for {}'.format(params['dataset_dir']))


def get_run_names(task_id, job_id, params, split_cnt):
    loc_feat = params['dataset']
    if params['dataset'] == 'mic':
        loc_feat = '{}_salsa'.format(params['dataset']) if params['use_salsalite'] else '{}_gcc'.format(params['dataset'])
    loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'
    unique_name = '{}_{}_{}_split{}_{}_{}'.format(task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat)
    best_any = os.path.join(params['model_dir'], '{}_model.h5'.format(unique_name))
    best_full = os.path.join(params['model_dir'], '{}_best_full_model.h5'.format(unique_name))
    return unique_name, best_any, best_full


def load_class_names(params):
    if params.get('class_mapping_path') and os.path.exists(params['class_mapping_path']):
        with open(params['class_mapping_path'], 'r') as handle:
            mapping = json.load(handle)
        return [mapping['class_id_to_name'][str(idx)] for idx in range(mapping['class_count'])]
    return [str(idx) for idx in range(params['unique_classes'])]


def write_metric_files(result_dir, summary, classwise_rows):
    with open(os.path.join(result_dir, 'metrics_summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)

    with open(os.path.join(result_dir, 'classwise_metrics.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class_id', 'class_name', 'ER', 'F', 'AngE_deg', 'DistE', 'RelDistE', 'LR', 'SELD'])
        writer.writerows(classwise_rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained dev checkpoint on the configured test split.')
    parser.add_argument('task_id', help='Parameter preset id, e.g. 231')
    parser.add_argument('job_id', help='Training job id used in train_seldnet.py')
    parser.add_argument('--ckpt', default='best_any', choices=['best_any', 'best_full'], help='Checkpoint alias to evaluate')
    parser.add_argument('--ckpt-path', default=None, help='Explicit checkpoint path; overrides --ckpt')
    parser.add_argument('--split-index', type=int, default=0, help='Split index to evaluate')
    parser.add_argument('--jackknife', action='store_true', help='Enable jackknife confidence intervals')
    args = parser.parse_args()

    params = parameters.get_params(args.task_id)
    if params['mode'] != 'dev':
        raise ValueError('This script only evaluates development/test splits.')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    tqdm.write('Using device: {}'.format(device))

    train_splits, val_splits, test_splits = get_dev_splits(params)
    split_cnt = args.split_index
    if split_cnt >= len(test_splits):
        raise IndexError('split-index {} out of range'.format(split_cnt))

    unique_name, best_any_path, best_full_path = get_run_names(args.task_id, args.job_id, params, split_cnt)
    ckpt_path = args.ckpt_path or (best_any_path if args.ckpt == 'best_any' else best_full_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt_path))

    tqdm.write('Run name: {}'.format(unique_name))
    tqdm.write('Checkpoint: {}'.format(ckpt_path))

    tqdm.write('Loading test dataset')
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
    )

    if params['modality'] == 'audio_visual':
        data_in, vid_data_in, data_out = data_gen_test.get_data_sizes()
        model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
    else:
        data_in, data_out = data_gen_test.get_data_sizes()
        model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

    tqdm.write('SELD-net data_in={} data_out={}'.format(data_in, data_out))
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)

    if params['multi_accdoa'] is True:
        criterion = seldnet_model.SEDDOAMultiTaskLoss(params) if params.get('explicit_sed_head') else seldnet_model.MSELoss_ADPIT(params)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(max(0, int(params.get('sed_warmup_epochs', 0))))
    else:
        criterion = nn.MSELoss()

    score_obj = ComputeSELDResults(params)
    result_dir = os.path.join(
        params['dcase_output_dir'],
        '{}_{}_test_eval_{}'.format(unique_name, os.path.splitext(os.path.basename(ckpt_path))[0], strftime('%Y%m%d%H%M%S', gmtime()))
    )
    cls_feature_class.delete_and_create_folder(result_dir)
    tqdm.write('Test predictions -> {}'.format(result_dir))

    test_loss = test_epoch(data_gen_test, model, criterion, result_dir, params, device, desc='Test(eval)')
    scores = score_obj.get_SELD_Results(result_dir, is_jackknife=args.jackknife)
    aux_metrics = score_obj.get_last_aux_metrics()

    if args.jackknife:
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = scores
        summary = {
            'checkpoint': ckpt_path,
            'test_loss': float(test_loss),
            'ER': test_ER[0],
            'F': test_F[0],
            'AngE_deg': test_LE[0],
            'DistE': test_dist_err[0],
            'RelDistE': test_rel_dist_err[0],
            'LR': test_LR[0],
            'SELD': test_seld_scr[0],
        }
        summary.update(aux_metrics)
        classwise_arrays = classwise_test_scr[0]
    else:
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = scores
        summary = {
            'checkpoint': ckpt_path,
            'test_loss': float(test_loss),
            'ER': float(test_ER),
            'F': float(test_F),
            'AngE_deg': float(test_LE),
            'DistE': float(test_dist_err),
            'RelDistE': float(test_rel_dist_err),
            'LR': float(test_LR),
            'SELD': float(test_seld_scr),
        }
        summary.update(aux_metrics)
        classwise_arrays = classwise_test_scr

    class_names = load_class_names(params)
    classwise_rows = []
    for class_id, class_name in enumerate(class_names):
        classwise_rows.append([
            class_id,
            class_name,
            float(classwise_arrays[0][class_id]),
            float(classwise_arrays[1][class_id]),
            float(classwise_arrays[2][class_id]),
            float(classwise_arrays[3][class_id]),
            float(classwise_arrays[4][class_id]),
            float(classwise_arrays[5][class_id]),
            float(classwise_arrays[6][class_id]),
        ])

    write_metric_files(result_dir, summary, classwise_rows)

    ang_acc = max(0.0, 1.0 - summary['AngE_deg'] / 180.0) if np.isfinite(summary['AngE_deg']) else np.nan
    tqdm.write('Test loss: {:.4f}'.format(summary['test_loss']))
    tqdm.write('ER/F/LR={:.3f}/{:.3f}/{:.3f}'.format(summary['ER'], summary['F'], summary['LR']))
    tqdm.write('AngE_deg/AngAcc={:.2f}/{:.3f}'.format(summary['AngE_deg'], ang_acc))
    tqdm.write('Dist/RelDist/SELD={:.2f}/{:.2f}/{:.2f}'.format(summary['DistE'], summary['RelDistE'], summary['SELD']))
    if aux_metrics.get('updown_total_gt', 0):
        tqdm.write(
            'UpDown Acc/F1/Prec/Rec={:.3f}/{:.3f}/{:.3f}/{:.3f} (n={})'.format(
                summary['updown_accuracy'],
                summary['updown_f1'],
                summary['updown_precision'],
                summary['updown_recall'],
                summary['updown_total_gt'],
            )
        )
    tqdm.write('Classwise results')
    tqdm.write('Class\tName\tER\tF\tAngE_deg\tDistE\tRelDistE\tLR\tSELD')
    for row in classwise_rows:
        tqdm.write('{}\t{}\t{:.3f}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(*row))
    tqdm.write('Saved summary -> {}'.format(os.path.join(result_dir, 'metrics_summary.json')))
    tqdm.write('Saved classwise -> {}'.format(os.path.join(result_dir, 'classwise_metrics.csv')))


if __name__ == '__main__':
    main()
