import pickle
from collections import defaultdict
import warnings
import json
from pathlib import Path
import time
import io

import torch
import pandas as pd
import numpy as np
import scipy
from pandas.io.formats.style_render import _escape_latex
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
    #print(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(df, f)


def save_preds(preds, save_path):
    with open(save_path, 'w') as f:
        json.dump(preds, f)

def set_test_metrics_columns(df, test_metrics=None):
    if test_metrics is None:
        test_metrics = ['coverage', 'wsc', 'cond_coverage_error_z', 'cond_coverage_error_repr', 'specialized_length', 'joint_length']#
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
        #df['train_size'] = float(df['model'].item().split('trainsize')[1][0:3])
        #df['model'] = df['model'].item().replace('_trainsize0.1', '').replace('_trainsize0.3', '').replace('_trainsize0.5', '').replace('_trainsize0.8', '')
        total_time = time.time() - start_time
        # if total_time > 1:
        #     print(f'Loading {path} took {total_time:.1f} seconds')
        #     print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = df.convert_dtypes()
    df.alpha = df.alpha.astype('float').round(8) # To avoid float precision issues
    df = df.set_index(['dataset', 'model', 'conformalizer', 'alpha', 'run_id', 'cal_size'])
    #df = df.set_index(['dataset', 'model', 'conformalizer', 'alpha', 'run_id', 'cal_size', 'train_size'])
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


def get_sequence_lengths(ds):
    dl = ds.get_dataloader(batch_size=64, shuffle=True)
    seq_len = [batch.mask.sum(dim=1).long() for batch in dl]
    seq_len = torch.concat(seq_len)
    return seq_len


def sort_by_dataset_size(df, ds_df, add_level=False):
    df_to_join = ds_df['Nb of instances', 'Training'].to_frame().droplevel(0, axis=1).rename_axis('dataset')
    if add_level:
        df_to_join.columns = pd.MultiIndex.from_product([df_to_join.columns, ['']])
    df = df.join(df_to_join).sort_values('Training', kind='stable')
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        df = df.drop(columns='Training')
    return df


def make_metric_df(df, alpha=None):
    if alpha is not None:
        df = df.query(f'alpha == {alpha}')
    metrics_series = df.metrics
    df = df.drop(columns=['metrics'])
    #df.columns = pd.MultiIndex.from_product([df.columns, ['']], names=['metric', 'is_conformal'])
    metrics = ['coverage', 'wsc', 'cond_coverage_error_z']
    for metric in metrics:
        df[metric] = metrics_series.map(lambda d: d[metric])
    df.columns.name = 'metric'
    df = df.stack(level='metric').rename('value')
    # Print duplicate entries in index
    df = df.unstack('conformalizer')
    return df


def sort_metric_table(pivot_df, ds_df):
    return sort_by_dataset_size(pivot_df, ds_df, add_level=True)


def pivot_table(metric_df, ds_df):
    pivot_df = metric_df.pivot_table(values='value', index=['dataset', 'model'], columns=['metric', 'conformalizer'], aggfunc='first')
    return sort_metric_table(pivot_df, ds_df)


def latex_style(df):
    df = df.copy()
    df.columns.names = list(map(_escape_latex, df.columns.names))
    return (df.style
            .format(precision=2)
            .format_index(escape='latex', axis=0)
            .format_index(escape='latex', axis=1)
    )


def formatted_table(metric_df, alpha, ds_df, print_latex=False):
    metric_df = metric_df.stack(level=[0, 1]).rename('value')
    index_names = metric_df.index.names
    num_df = metric_df.reset_index()
    coverage_mask = num_df['metric'] == 'coverage'
    num_df.loc[coverage_mask, 'value'] = (num_df.loc[coverage_mask, 'value'] - (1 - alpha)).abs()
    num_df = num_df.set_index(index_names)['value']
    # `ranks` contains the rank, 1-based indexed, per dataset and metric
    # In case of equality, the same rank is given (given by method='min')
    ranks = num_df.groupby(['dataset', 'metric', 'model']).rank(method='min')
    # `num_df[bold_mask]` contains the values that should be bolded
    bold_mask = num_df.index.isin(ranks[ranks == 1].index)
    bold_mask = pd.Series(data=bold_mask, index=num_df.index)
    # `num_df[italic_mask]` contains the values that should be italic
    italic_mask = num_df.index.isin(ranks[ranks == 2].index)
    italic_mask = pd.Series(data=italic_mask, index=num_df.index)
    # We build the pivot table in the same way for the values and for the mask
    pivot_df = pivot_table(metric_df.to_frame(), ds_df)
    pivot_bold_mask_df = pivot_table(bold_mask.rename('value').to_frame(), ds_df)
    pivot_italic_mask_df = pivot_table(italic_mask.rename('value').to_frame(), ds_df)
    # The length can sometimes be NaN. This propagates in the mask
    pivot_bold_mask_df = pivot_bold_mask_df.fillna(False)
    pivot_italic_mask_df = pivot_italic_mask_df.fillna(False)
    # `css_bold` contains the CSS properties of the final table 
    css = pivot_bold_mask_df.copy().astype('str')
    css[:] = ''
    css[pivot_bold_mask_df] += 'font-weight: bold;'
    css[pivot_italic_mask_df] += 'font-style: italic;'
    
    if print_latex:
        latex_styler = latex_style(pivot_df)
        latex_styler.apply(lambda df: css, axis=None)
        print(latex_styler.to_latex(hrules=True, multicol_align='c', convert_css=True))
    
    styler = pivot_df.style
    return styler.apply(lambda df: css, axis=None).format(precision=3)
