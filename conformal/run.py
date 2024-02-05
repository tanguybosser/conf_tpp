import json
import os
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Subset

from tpps.models import get_model
from tpps.utils.cli import parse_args
from tpps.utils.data import get_loader, load_data
from tpps.utils.run import make_deterministic

from .conformalizer import (
    APS,
    CHDR,
    CHDR_RAPS,
    CPROB,
    CQR,
    CQRL,
    CQRL_RAPS,
    HDR,
    QR,
    QRL,
    RAPS,
    CHDRTime,
    ConformalConstantTime,
    HDRTime,
    Naive_HDR_RAPS,
    Naive_QRL_RAPS,
    NaiveAPS,
    NaiveRAPS,
)
from .dataframes import load_df, save_runs
from .run_config import RunConfig
from .tester import Tester

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*input value tensor is non-contiguous.*',
)

conformalizer_builders_time = {
    'C-Const': ConformalConstantTime,
    'C-QRL': CQRL,
    'N-QRL': QRL,
    'C-QR': CQR,
    'N-QR': QR,
    'N-HDR-T': HDRTime,
    'C-HDR-T': CHDRTime,
}

conformalizer_builders_mark = {
    'C-RAPS': RAPS,
    'N-RAPS': NaiveRAPS,
    'C-APS': APS,
    'N-APS': NaiveAPS,
    'C-PROB': CPROB,
}

conformalizer_builers_joint = {
    'C-QRL-RAPS': CQRL_RAPS,
    'N-QRL-RAPS': Naive_QRL_RAPS,
    'C-HDR-RAPS': CHDR_RAPS,
    'N-HDR-RAPS': Naive_HDR_RAPS,
    'C-HDR': CHDR,
    'N-HDR': HDR,
}

conformalizer_builders = {
    **conformalizer_builders_time,
    **conformalizer_builders_mark,
    **conformalizer_builers_joint,
}


def run_conformalizer(rc, model, dl_calib, dl_test, args):
    conformalizer = conformalizer_builders[rc.conformalizer](dl_calib, model, rc.alpha, args)
    print(f'q = {conformalizer.q}', flush=True)
    args.conformalizer = rc.conformalizer
    metrics = Tester(dl_test, dl_calib, model, rc.alpha, conformalizer, rc, args).compute_test_metrics()
    metrics['q'] = conformalizer.q
    metrics['scores'] = conformalizer.scores
    return metrics


def run_conformalizer_and_save(rc, model, loaders, args):
    path = rc.get_path(args.save_results_conf_dir)
    if path.exists():
        print(f'Already exists: {path}')
        return
    print(rc.summary_str())
    metrics = run_conformalizer(rc, model, loaders['cal'], loaders['test'], args)
    rc.metrics = metrics
    save_runs([rc], path)


def run_model(args):
    args.save_results_conf_dir = Path('results_conf') / args.exp_dir
    
    assert args.model_name_short is not None
    datasets = load_data(args=args)
    if args.limit_cal_size is not None:
        subset_indices = list(range(args.limit_cal_size))
        datasets['cal'] = Subset(datasets['cal'], subset_indices)
    loaders = {
        'cal': get_loader(datasets['cal'], args=args, shuffle=False),
        'test': get_loader(datasets['test'], args=args, shuffle=False),
    }
    model = load_model(args)
    args.include_poisson = False if not 'poisson' in args.model_name_short else True
    print(f'====== Model: {args.model_name_short} ======')
    for alpha in args.alphas_cal:
        print(f'=== alpha: {alpha} ===')
        alpha = np.round(alpha, 2)
        print('Conformalizers', args.conformalizers)
        for conformalizer_name in args.conformalizers:
            print(f'--- conformalizer: {conformalizer_name} ---')
            if conformalizer_name.startswith('N-'):
                model_name_long = args.model_name.replace('conformal', 'trainingcal')
                model = load_model(args, model_name=model_name_long)
            rc = RunConfig(
                args.dataset,
                args.model_name_short,
                conformalizer_name,
                alpha,
                args.split,
                args.limit_cal_size,
            )
            run_conformalizer_and_save(rc, model, loaders, args)


def load_model(args, model_name=None):
    if model_name is not None:
        model_path = os.path.join(args.save_check_dir, model_name + '.pth')
    else:
        model_path = os.path.join(args.save_check_dir, args.model_name + '.pth')
    model_args = load_model_args(args)
    args.include_poisson = model_args.include_poisson
    model = get_model(model_args)
    model.load_state_dict(th.load(model_path, map_location=args.device))
    return model 


def load_model_args(args: Namespace):
    args_path = os.path.join(args.save_check_dir, 'args')
    args_path = os.path.join(args_path, args.model_name + '.json')
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    model_args = Namespace(**model_args)
    model_args.decoder_mc_prop_est = 250
    model_args.device = args.device
    return model_args


if __name__ == '__main__':
    parsed_args = parse_args()
    json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
    json_dir = os.path.join(json_dir, parsed_args.dataset)
    json_dir = os.path.join(json_dir, f'split_{parsed_args.split}')    
    json_path = os.path.join(json_dir, 'args.json')
    with open(json_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(parsed_args)
    args_dict.update(args_dict_json)
    parsed_args = Namespace(**args_dict) 
    make_deterministic(seed=parsed_args.seed)
    parsed_args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    run_model(parsed_args)
