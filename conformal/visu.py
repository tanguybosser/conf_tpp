import json
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import seaborn as sns
import numpy as np
import pickle as pkl
import os 
import pandas as pd

from .plot import map_dataset, map_model

results_dir = '../results_conf'
preds_dir = '../preds'
data_dir = '../data/baseline3'


def show_intervals_on_axis(fig, axis, pred):
    n_marks = 50
    axis.set_ylim([-0.5, n_marks-0.5])
    length = 0
    for mark, pred_mark in pred.items():
        for interval in pred_mark:
            left, right = interval
            length += right - left
            axis.hlines(int(mark), left, right, color='red', lw=4)
    print(length)

    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.04, 0.5, 'k', va='center', rotation='horizontal', fontsize=16)
    axis.set_xlabel(r'$|\hat{R}_\tau(X_{n+1})|$', fontsize=16)


def show_intervals(dataset, model, conformalizers, split, alpha,
                   n_marks, n_plots=20):
    all_preds, results = load_preds_results(dataset, model, conformalizers, split, alpha, n_plots)
    for i in range(n_plots):
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(2,2, figsize=(8,8), sharex='row', sharey=True)
        mpl.rcParams.update(mpl.rcParamsDefault)
        for k, conformalizer in enumerate(conformalizers):
            idx1 = int(k/(len(conformalizers)/2))
            idx2 = int(k % (len(conformalizers)/2))
            ax[idx1,idx2].set_ylim([-0.5,n_marks-0.5])
            coverage = results[results['conformalizer'] == conformalizer]['metrics'].item()['coverage']
            coverage = str(np.round(coverage, 2))
            length = 0
            for mark, cov in all_preds[k][i].items():
                if len(cov) != 0:
                    lb = cov[0][0]
                    ub = cov[0][1]
                    length += ub -lb 
                    ax[idx1,idx2].hlines(int(mark), lb, ub , color='red', lw=4)
            ax[idx1, idx2].set_title(f'{conformalizer} ({coverage}/{np.round(length,2)})', fontsize=14)
            ax[idx1, idx2].yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.text(0.04, 0.5, 'k', va='center', rotation='horizontal', fontsize=16)
            fig.text(0.5, 0.04, r'$|\hat{R}_\tau(X_{n+1})|$', ha='center', fontsize=16)
            fig.savefig(f'../figures/intervals/{dataset}_{model}_{i}.png', bbox_inches='tight')
            plt.close()

def load_preds_results(dataset, model, conformalizers, split, alpha, n_plots):
    all_preds = defaultdict()
    file_names = [os.path.join(preds_dir, f'preds_{dataset}_{model}_split{split}_{conformalizer}_{alpha}.json')
                 for conformalizer in conformalizers]
    results = os.path.join(results_dir, f'conformal_{dataset}_{model}_split{split}_{alpha}.pkl')
    for k, file_name in enumerate(file_names):
        with open(file_name, 'r') as f:
            preds = json.load(f)
        all_preds[k] = preds[:n_plots]
    with open(results, 'rb') as f:
        results = pkl.load(f)    
    return all_preds, results

def tabular_results(models, datasets, n_split, results_dir, alpha, std=True):
    files = os.listdir(results_dir)
    conformalizers = [
        'C-ConstAll',
        'C-ConstAPS',
        'C-CQRAll',
        'Naive',
        'C-Naive',
        'HDR',
        'C-HDR'
    ]
    all_results = {
        'coverage':{conformalizer:[] for conformalizer in conformalizers},
        'length':{conformalizer:[] for conformalizer in conformalizers}, 
        'wsc':{conformalizer:[] for conformalizer in conformalizers}
    }
    for model in models:
        for dataset in datasets:
            coverages = {conformalizer:[] for conformalizer in conformalizers}
            lengths = {conformalizer:[] for conformalizer in conformalizers}
            wscs = {conformalizer:[] for conformalizer in conformalizers}
            for split in range(n_split):
                file_name = None
                if 'mlp-cm' in model:
                    file_to_find = f'conformal_poisson_{dataset}_{model}_split{split}_{alpha}.pkl'
                else:
                    file_to_find = f'conformal_{dataset}_{model}_split{split}_{alpha}.pkl'
                for file in files:
                    if file.startswith(file_to_find):
                        file_name = file 
                        break 
                if file_name is None:
                    raise ValueError('File not found')
                file_name = os.path.join(results_dir, file_name)
                with open(file_name, 'rb') as f:
                    results = pkl.load(f)
                for conformalizer in conformalizers:
                    coverages[conformalizer].append(results[results['conformalizer'] == conformalizer]['metrics'].item()['coverage'])
                    lengths[conformalizer].append(results[results['conformalizer'] == conformalizer]['metrics'].item()['length'])
                    wscs[conformalizer].append(results[results['conformalizer'] == conformalizer]['metrics'].item()['wsc'])
            for conformalizer in conformalizers:
                mean_cov = str(np.round(np.mean(coverages[conformalizer]),3))
                ste_cov = str(np.round(np.std(coverages[conformalizer])/np.sqrt(len(coverages[conformalizer]))))
                mean_length = str(np.round(np.mean(lengths[conformalizer]),3))
                ste_length = str(np.round(np.std(lengths[conformalizer])/np.sqrt(len(lengths[conformalizer])),3))
                mean_wsc = str(np.round(np.mean(wscs[conformalizer]),3))
                ste_wsc = str(np.round(np.std(wscs[conformalizer])/np.sqrt(len(wscs[conformalizer])),3))
                if std:
                    mean_cov = f'{mean_cov} ({ste_cov})'
                    mean_length = f'{mean_length} ({ste_length})'
                    mean_wsc = f'{mean_wsc} ({ste_wsc})'
                else:
                    mean_cov = f'{mean_cov}'
                    mean_length = f'{mean_length}'
                    mean_wsc = f'{mean_wsc}'
                all_results['coverage'][conformalizer].append(mean_cov)
                all_results['length'][conformalizer].append(mean_length)
                all_results['wsc'][conformalizer].append(mean_wsc)
    models = [map_model(model) for model in models]
    datasets = [map_dataset(dataset) for dataset in datasets]
    for metric, metrics_dic in all_results.items():
        cols, data = [], []
        for conformalizer, val_list in metrics_dic.items():
            cols.append((metric, conformalizer))
            data.append(val_list)
        df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
        models_datasets = []
        for model in models:
            for dataset in datasets:
                models_datasets.append((model, dataset))
        df.insert(0,'', models_datasets)
        columns = 'c' * df.shape[1]
        df = df.astype('str')
        df_tex = df.to_latex(index=False, escape=False, multicolumn_format='c', column_format=columns)
        print(df_tex)

def interval_lengthts_distribution(dataset, methods):
    fig, ax = plt.subplots()
    all_lengths = []
    for method in methods:
        lengths = []
        file_name = f'../preds_{dataset}_{method}.json'
        with open(file_name, 'r') as f:
            preds = json.load(f)
        for pred in preds:
            length = 0
            for mark_region in pred.values():
                for interval in mark_region:
                    left, right = interval
                    length += right - left
            lengths.append(length)
        all_lengths.append(lengths)
        print(f'Mean length for method {method}: {np.mean(lengths)}')
        print(f'Median length for method {method}: {np.median(lengths)}')
    ax.boxplot(all_lengths)
    ax.set_ylabel(r'$|\hat{R}_\tau(X_{n+1})|$', fontsize=24)
    #methods = [map_method_name(method) for method in methods]
    ax.set_xticks([1, 2, 3, 4], methods)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(map_dataset(dataset), fontsize=20)
    fig.savefig(f'../figures/dist/{dataset}.png', bbox_inches='tight')

def window_length_dis(dataset):
    data_path = os.path.join(data_dir, dataset)
    data_path  = os.path.join(data_path, dataset + '.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    num_points = [len(seq) for seq in data]
    window = [seq[-1]['time'] for seq in data]
    fig, ax = plt.subplots() 
    ax.scatter(window, num_points, s=5, label=map_dataset(dataset))
    ax.set_ylabel(r'$N(t_n)$', fontsize=20)
    ax.set_xlabel(r'$t_n$', fontsize=20)  
    ax.legend()
    fig.savefig(f'../figures/scatterplots/{dataset}.png', bbox_inches='tight')

def map_method_name(method):
    if 'Non-conformal Joint Naive' in method:
        mapping = 'Naive'
    elif 'Joint Naive' in method:
        mapping = 'C-Naive'
    elif 'Non-conformal HPD-split' in method:
        mapping = 'HDR'
    elif 'HPD-split' in method:
        mapping = 'C-HDR'        
    return mapping




if __name__ == "__main__":
    datasets = {
                'lastfm_filtered':50,
                'mooc_filtered':50,
                #'github_filtered':8,
                'reddit_filtered_short':50,
                'retweets_filtered_short':3,
                'stack_overflow_filtered':22
    }
    models = [
        'gru_cond-log-normal-mixture_temporal_with_labels_conformal'
    ]
    conformalizers = [
        'Naive',
        'C-Naive',
        'HDR',
        'C-HDR'
    ]
    split = 0
    model = 'gru_cond-log-normal-mixture_temporal_with_labels_conformal'
    alpha = 0.2
    n_marks = 50
    '''
    for dataset, n_marks in datasets.items():
        show_intervals(dataset, model, conformalizers, split, alpha, n_marks, n_plots=10)  
    '''
    #for dataset in datasets:
    #    interval_lengthts_distribution(dataset, methods)
    '''
    for dataset in datasets.keys():
        window_length_dis(dataset)    
    '''
    n_split = 5
    results_dir = '../results_conf'
    alpha = 0.2
    models = [
        'gru_cond-log-normal-mixture_temporal_with_labels_conformal'
    ]
    tabular_results(models, datasets, n_split, results_dir, alpha, std=False)