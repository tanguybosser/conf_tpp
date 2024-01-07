# import sys, os
# sys.path.append(os.path.abspath(os.path.join('..', 'cntpp')))
# sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))

import json
import numpy as np
import os
import torchvision  
import pickle as pkl
from datetime import datetime
import warnings

import torch as th
from torch.utils.data import DataLoader, Subset
import pandas as pd

from argparse import Namespace
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tpps.utils.events import get_events, get_window

from tpps.models import get_model
from tpps.models.base.process import Process
from tpps.utils.cli import parse_args
from tpps.utils.data import get_loader, load_data
from tpps.utils.run import make_deterministic

import torch


from .run_config import RunConfig
from .conformalizer import (
    ConformalConstantTime,
    CQRL,
    QRL,
    CQR,
    QR, 
    CHDRTime,
    HDRTime,
    RAPS,
    NaiveRAPS,
    APS, 
    NaiveAPS,
    APS2, 
    NaiveAPS2,
    ConformalizedProb, 
    ConformalizedProb2,
    CQRL_RAPS,
    Naive_QRL_RAPS,
    CHDR_RAPS,
    Naive_HDR_RAPS,
    CHDR,
    HDR,
)
from .tester import Tester
from .dataframes import save_runs, load_df, make_metric_df


torchvision.__version__ = '0.4.0'
warnings.filterwarnings("ignore", category=UserWarning, message=".*input value tensor is non-contiguous.*")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


datasets_list = [
    'lastfm_filtered'
    'mooc_filtered'
    'github_filtered'
    'reddit_filtered_short'
    'retweets_filtered_short'
    'stack_overflow_filtered'
]

models_list = [
    'gru_cond-log-normal-mixture_temporal_with_labels_conformal'
    'gru_rmtpp_temporal_with_labels_conformal'
    'gru_thp_temporal_with_labels_conformal'
    'gru_mlp-cm_learnable_with_labels_conformal'
]

# ====== 1st possibility ======

#Time: QR, QRL, HDR-T, C-QR, C-QRL, C-HDR-T
#Mark: APS, RAPS, C-APS, C-RAPS, C-PMF
#Joint: IND, HDR, C-IND, (C-PDF), C-HDR

# (IND = CQRL-RAPS)


# ====== 2nd possibility ======

#Time: N-CQR, N-CQRL, N-HPD-split-T, CQR, CQRL, HPD-split-T
#Mark: N-APS, N-RAPS, APS, RAPS, PMF
#Joint: N-CQRL-RAPS, N-HPD-split, CQRL-RAPS, (PDF), HPD-split


# ====== 3rd possibility ======

#Time: N-QR, N-QRL, N-HDR, C-QR, C-QRL, C-HDR
#Mark: N-APS, N-RAPS, C-APS, C-RAPS, C-PROB
#(Mark: N-HDR, N-RHDR, C-HDR, C-RHDR, C-PROB)
#Joint: N-IND, N-HDR, C-IND, (C-PROB), C-HDR

# (IND = QRL-RAPS)
# (C-IND = C-QRL-RAPS)

# (IND = HDR-RAPS)
# (C-IND = C-HDR-RAPS)


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
    
    'C-APS2': APS2,
    'N-APS2': NaiveAPS2,

    'C-PROB': ConformalizedProb,
    'C-PROB2': ConformalizedProb2

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
    **conformalizer_builers_joint
}

conformalizer_builders_oracle = {
     'Naive': Naive_HDR_RAPS,
     'HDR': HDR,
}


def run_conformalizer(rc, model, dl_calib, dl_test, args):
    #print(sum(batch.inter_times.shape[0] for batch in dl_test), flush=True)
    conformalizer = conformalizer_builders[rc.conformalizer](dl_calib, model, rc.alpha, args)
    print(f'q = {conformalizer.q}', flush=True)
    # plot_scores(conformalizer.scores, conformalizer.q, rc.alpha)
    # savefig(rc.get_dir_path('figures/test') / 'scores.pdf')
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
    model_path = os.path.join(args.save_check_dir, args.model_name + '.pth')
    model_name  = args.model_name_short
    model_name_long = args.model_name 
    assert model_name is not None
    args = load_args(args)
    datasets = load_data(args=args)
    if args.limit_cal_size is not None:
        subset_indices = list(range(args.limit_cal_size))
        datasets['cal'] = Subset(datasets['cal'], subset_indices)
    loaders = {
        'cal': get_loader(datasets['cal'], args=args, shuffle=False),
        'test': get_loader(datasets['test'], args=args, shuffle=False)
    }
    args.decoder_mc_prop_est = 250
    model = get_model(args)
    model.load_state_dict(th.load(model_path, map_location=args.device))
    num_params = count_parameters(model)
    print(f'Model instantiated ({args.encoder_encoding}/{args.encoder}/{args.dataset}/{num_params})')
    print(f'====== Model: {model_name} ======')
    #for alpha in np.arange(0.1, 1.0, 0.1):
    '''
    if 'hawkes' not in args.model_name_short:
        conformalizers = conformalizer_builders
    else:
        conformalizers = conformalizer_builders_oracle
    '''
    for alpha in args.alphas_cal:
        print(f'=== alpha: {alpha} ===')
        alpha = np.round(alpha, 2)
        print('Conformalizers', args.conformalizers)
        for conformalizer_name in args.conformalizers:
            print(f'--- conformalizer: {conformalizer_name} ---')
            if conformalizer_name.startswith('N-'):
                model_name_long = model_name_long.replace('conformal', 'trainingcal')
                model_path = os.path.join(args.save_check_dir, model_name_long + '.pth')
                model.load_state_dict(th.load(model_path, map_location=args.device))
            rc = RunConfig(
                args.dataset, model_name, conformalizer_name, alpha, args.split, args.limit_cal_size
            )
            run_conformalizer_and_save(rc, model, loaders, args)
        # df = load_df(args.save_results_conf_dir / rc.model / rc.dataset / str(rc.alpha))
        # df = make_metric_df(df)
        # with pd.option_context(
        #     'display.max_rows', None, 
        #     'display.max_columns', None, 
        #     'max_colwidth', 50, 
        #     'expand_frame_repr', False,
        #     'display.precision', 3,
        # ):
        #     print(df, flush=True)


def load_args(args:Namespace):
    batch_size = args.batch_size
    load_dir = args.load_from_dir
    save_dir = args.save_results_dir
    alphas_cal = args.alphas_cal
    model_name_short = args.model_name_short 
    args_path = os.path.join(args.save_check_dir, 'args')
    args_path = os.path.join(args_path, args.model_name + '.json')
    k_reg = args.k_reg
    n_partitions = args.n_partitions
    exp_dir = args.exp_dir
    lambda_reg = args.lambda_reg
    conformalizers = args.conformalizers
    eval_cond_coverage = args.eval_cond_coverage
    args = vars(args)
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    args.update(model_args)
    args = Namespace(**args)
    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        args.device = th.device('cuda')
    else:
        args.device = th.device('cpu')
    print(args.device)
    args.batch_size = batch_size
    args.load_from_dir = load_dir
    args.save_results_dir = save_dir
    args.alphas_cal = alphas_cal
    args.model_name_short = model_name_short
    args.k_reg = k_reg
    args.n_partitions = n_partitions
    args.exp_dir = exp_dir
    args.lambda_reg = lambda_reg
    args.conformalizers = conformalizers
    args.eval_cond_coverage = eval_cond_coverage
    #current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.save_results_conf_dir = Path('results_conf') / args.exp_dir
    return args 


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.load_from_dir is not None:
        json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
        json_dir = os.path.join(json_dir, parsed_args.dataset)
        if parsed_args.split is not None:
            split = 'split_{}'.format(parsed_args.split)
            json_dir = os.path.join(json_dir, split)
        json_path = os.path.join(json_dir, 'args.json')
        with open(json_path, 'r') as fp:
            args_dict_json = json.load(fp)
        args_dict = vars(parsed_args)
        shared_keys = set(args_dict_json).intersection(set(args_dict))
        for k in shared_keys:
            v1, v2 = args_dict[k], args_dict_json[k]
            is_equal = np.allclose(v1, v2) if isinstance(
                v1, np.ndarray) else v1 == v2
            if not is_equal:
                print(f"    {k}: {v1} -> {v2}", flush=True)
        args_dict.update(args_dict_json)
        parsed_args = Namespace(**args_dict)
        parsed_args.mu = np.array(parsed_args.mu, dtype=np.float32)
        parsed_args.alpha = np.array(
            parsed_args.alpha, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)
        parsed_args.beta = np.array(
            parsed_args.beta, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)

    else:
        parsed_args.data_dir = os.path.expanduser(parsed_args.data_dir)
        parsed_args.save_dir = os.path.join(parsed_args.data_dir, "None")
        Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)
    make_deterministic(seed=parsed_args.seed)
    parsed_args.device = 'cuda' if th.cuda.is_available() else 'cpu'
    run_model(parsed_args)
