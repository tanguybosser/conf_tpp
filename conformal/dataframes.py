import io
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from tqdm.auto import tqdm


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def op_without_index(df, op):
    names = df.index.names
    df = op(df.reset_index())
    if names != [None]:
        df = df.set_index(names)
    return df


def save_runs(runs, save_path):
    series = [rc.to_series() for rc in runs]
    df = pd.concat(series, axis=1).T
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # print(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(df, f)


def set_test_metrics_columns(df, test_metrics=None):
    if test_metrics is None:
        test_metrics = [
            'coverage',
            'specialized_length',
            'joint_length',
            'geom_specialized_length',
            'wsc',
            'cond_coverage_error_z',
            'cond_coverage_error_repr',
        ]
    for metric in test_metrics:
        df[metric] = df.apply(lambda df: df.metrics.get(metric, None), axis=1)


def load_df(save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    dfs = []
    paths = list(save_path.rglob('*.pkl'))
    pbar = tqdm(paths)
    for path in pbar:
        pbar.set_description(f'Loading {path}')
        start_time = time.time()
        with open(path, 'rb') as f:
            df = CPU_Unpickler(f).load()
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = df.convert_dtypes()
    df.alpha = df.alpha.astype('float').round(8)   # To avoid float precision issues
    df['type'] = df.apply(lambda df: df.metrics['type'], axis=1)
    df = df.set_index(
        [
            'dataset',
            'model',
            'conformalizer',
            'type',
            'alpha',
            'run_id',
            'cal_size',
        ]
    )
    return df


def agg_mean_std(x):
    mean = np.mean(x)
    std = None
    if len(x) > 1:
        std = scipy.stats.sem(x, ddof=1)
    return (mean, std)


def format_cell(x):
    mean, sem = x
    if np.isnan(mean):
        return 'NA'
    s = f'{mean:#.3}'
    if sem is not None:
        sem = float(sem)
        s += rf' +- {sem:#.2}'
    return s


def agg_mean_std_format(x):
    return format_cell(agg_mean_std(x))


def make_df_mean(df, agg='mean'):
    df = df.reset_index(level='run_id', drop=True)
    agg_fn = {
        'mean': 'mean',
        'mean_std': agg_mean_std_format,
    }[agg]
    return df.groupby(df.index.names, dropna=False).agg(agg_fn)
