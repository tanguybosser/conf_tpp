import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .dataframes import make_df_mean


def get_conformalizers_order_joint():
    return [
        'N-QRL-RAPS',
        'N-HDR-RAPS',
        'N-HDR',
        'C-QRL-RAPS',
        'C-HDR-RAPS',
        'C-HDR',
    ]


def get_conformalizers_order_time():
    return ['N-QR', 'N-QRL', 'N-HDR-T', 'C-Const', 'C-QR', 'C-QRL', 'C-HDR-T']


def get_conformalizers_order_mark():
    return ['N-APS', 'N-RAPS', 'C-PROB', 'C-APS', 'C-RAPS']


def rename_conformalizers(c):
    if c.startswith('N-'):
        c = 'H-' + c[2:]
    d = {
        'H-HDR-T': 'H-HDR',
        'C-HDR-T': 'C-HDR',
    }
    if c in d:
        c = d[c]
    return c


def get_conformalizers_order(type):
    if type == 'joint':
        order = get_conformalizers_order_joint()
    elif type == 'time':
        order = get_conformalizers_order_time()
    else:
        order = get_conformalizers_order_mark()
    order = [rename_conformalizers(c) for c in order]
    return order


def rename_and_filter_conformalizers(df, type):
    df = df.copy()
    df['conformalizer'] = df['conformalizer'].apply(rename_conformalizers)
    conformalizer_order = get_conformalizers_order(type)
    df = df.query('conformalizer in @conformalizer_order')
    return df


def map_metric(metric):
    metrics = {
        'coverage': 'MC',
        'wsc': 'WSC',
        'specialized_length': 'Length',
        'relative_length': 'R. Length',
        'geom_specialized_length': 'G. Length',
        'cond_coverage_error_z': 'CCE',
        'cond_coverage_error_repr': 'Cond. cov. X',
    }
    return metrics[metric]


def map_model(model):
    mapping = {
        'gru_cond-log-normal-mixture_temporal_with_labels_conformal': 'CLNM',
        'gru_cond-log-normal-mixture_temporal_with_labels_trainingcal': 'CLNM',
        'gru_cond-log-normal-mixture_temporal_with_labels': 'CLNM',
        'poisson_gru_rmtpp_temporal_with_labels_conformal': 'RMTPP',
        'poisson_gru_rmtpp_temporal_with_labels_trainingcal': 'RMTPP',
        'poisson_gru_thp_temporal_with_labels_conformal': 'THP',
        'poisson_gru_thp_temporal_with_labels_trainingcal': 'THP',
        'poisson_gru_mlp-cm_learnable_with_labels_conformal': 'FNN',
        'poisson_gru_mlp-cm_learnable_with_labels_trainingcal': 'FNN',
        'poisson_gru_sahp_temporal_with_labels_conformal': 'SAHP',
        'poisson_gru_sahp_temporal_with_labels_trainingcal': 'SAHP',
        'identity_poisson_times_only': 'Poisson',
        'stub_hawkes_fixed': 'Hawkes',
    }
    return mapping[model]


def map_dataset(dataset):
    mapping = {
        'lastfm_filtered': 'LastFM',
        'mooc_filtered': 'MOOC',
        'github_filtered': 'Github',
        'reddit_filtered_short': 'Reddit',
        'retweets_filtered_short': 'Retweets',
        'stack_overflow_filtered': 'Stack Overflow',
        'hawkes_exponential_mutual': 'Hawkes',
    }
    return mapping[dataset]


def get_datasets_order():
    return ['LastFM', 'MOOC', 'Reddit', 'Retweets', 'Stack Overflow']
    # return ['Hawkes']


def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)


def set_notebook_options():
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 80)
    pd.set_option('display.float_format', '{:.3f}'.format)
    # Avoid the SettingWithCopyWarning, which can be triggered even if there is no problem.
    # Just be extra careful when dealing with functions that can return a copy of a dataframe.
    pd.options.mode.chained_assignment = None
    mpl.rcParams['axes.formatter.limits'] = (-2, 4)
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    # sns.set_theme()


def plot_scores(scores, q, alpha):
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    scores = scores.sort().values.detach().cpu()
    ax.plot(scores, torch.linspace(0, 1, scores.shape[0]), label='$F_S$')
    ax.axvline(
        q.detach().cpu(),
        ymin=0,
        ymax=1 - alpha,
        label=r'$\hat{q}$',
        color='r',
        linestyle='--',
    )
    ax.axhline(
        1 - alpha,
        xmin=0,
        xmax=q.detach().cpu().item(),
        label=r'$1 - \alpha$',
        color='r',
        linestyle='--',
    )
    ax.legend()
    return fig


def plot_joint_densities(self, preds, past_events, target_time, target_label):
    mpl.rcParams.update(mpl.rcParamsDefault)
    batch_shape = past_events.times.shape[:1]

    # last_event = past_events.times_final
    # epsilon, step = 1e-7, 0.000001
    # queries = torch.stack([torch.arange(float(event)+epsilon, float(event)+0.0004, step)
    #                        for i, event in enumerate(last_event)], dim=0).to(self.args.device)

    nb_steps = 500
    alpha = torch.linspace(0.01, 0.7, nb_steps, device=self.args.device)[None, :].expand(
        batch_shape + (nb_steps,)
    )
    queries = self.icdf(alpha, past_events)

    log_density, _, _ = self.model.log_density(query=queries, events=past_events)

    densities = log_density.detach().exp()
    # densities = log_densities.exp()
    queries = queries.detach()

    for j in range(densities.shape[0]):
        nb_marks_to_plot = densities.shape[2]
        fig, ax = plt.subplots(nb_marks_to_plot, 1, sharex=True, sharey=True, figsize=(4, 40))
        pred = preds[j]
        query = queries[j]
        log_density = densities[j]
        for i in range(nb_marks_to_plot):
            pred_mark = pred[i]
            log_density_mark = log_density[:, i]
            ax[i].plot(query, log_density_mark)
            for interval in pred_mark:
                left, right = interval
                mask = (left <= query) & (query <= right)
                x = query[mask]
                y2 = log_density_mark[mask]
                y1 = torch.zeros_like(y2)
                ax[i].fill_between(x=x, y1=y1, y2=y2, alpha=0.2, color='red')
        target_label_index = target_label.argmax(-1)
        ax[target_label_index[j]].axvline(x=target_time[j].item(), color='green')
        fig.savefig(f'figures/intervals/test_{j}.pdf', bbox_inches='tight')


def generate_color_palette(conformalizer_order):
    unique = [c[2:] for c in conformalizer_order]
    unique = sorted(set(unique), key=unique.index)   # Remove duplicates but keep order
    colors = sns.color_palette()
    color_map = {u: color for u, color in zip(unique, colors)}
    return {c: color_map[c[2:]] for c in conformalizer_order}


def add_hatches(g, conformalizer_order, n_groups):
    # We have to add the hatches manually because seaborn does not support it
    patch_order = []
    for c in conformalizer_order:
        patch_order += [c] * n_groups
    patch_order += conformalizer_order
    for c, bar in zip(patch_order, g.patches):
        if not c.startswith('C-'):
            bar.set_hatch(r'\\')
            bar.set_edgecolor('white')


def plot_metrics_on_real_world_datasets(df, alpha, x='dataset', type='joint'):
    # One dataset per column and one metric per row
    df = df.reset_index()
    df = rename_and_filter_conformalizers(df, type)
    conformalizer_order = get_conformalizers_order(type)

    df = df.query('alpha == @alpha')
    x_values = df[x].unique()
    if type in ['time', 'joint']:
        metrics = [
            'coverage',
            'relative_length',
            'gmean_length',
            'wsc',
            'cond_coverage_error_z',
        ]
    else:
        metrics = [
            'coverage',
            'specialized_length',
            'gmean_length',
            'wsc',
            'cond_coverage_error_z',
        ]
    conformalizers = df.conformalizer.unique()
    nrows, ncols = len(metrics), 1

    df['dataset'] = pd.Categorical(df['dataset'], categories=get_datasets_order(), ordered=True)
    x_size = len(x_values) * len(conformalizers) ** 0.5 * 0.5
    # x_size = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(x_size, nrows * 1), squeeze=False, sharex=True)
    axes = axes[:, 0]

    color_palette = generate_color_palette(conformalizer_order)

    for i, metric in enumerate(metrics):
        axis = axes[i]
        g = sns.barplot(
            data=df,
            x=x,
            y=metric,
            hue='conformalizer',
            hue_order=conformalizer_order,
            palette=color_palette,
            ax=axis,
        )
        add_hatches(g, conformalizer_order, len(x_values))

        g.legend().remove()
        axis.set_xlabel(None)
        axis.set_ylabel(map_metric(metric))
        if metric in ['coverage', 'wsc']:
            axis.set_ylim(0, 1)
            axis.axhline(y=1 - alpha, color='black', linestyle='--')
        if (
            metric
            in [
                'length',
                'relative_length',
                'cond_coverage_error_z',
                'cond_coverage_error_repr',
            ]
            and type != 'mark'
        ):
            # axis.set_ylim(bottom=0)
            axis.set_yscale('log')

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.97),
        frameon=True,
        ncol=7,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_metrics_on_real_world_datasets_with_separate_axes(df, alpha, x='dataset', type='joint'):
    # One dataset per column and one metric per row
    df = df.reset_index()
    df = rename_and_filter_conformalizers(df, type)
    conformalizer_order = get_conformalizers_order(type)
    df['dataset'] = pd.Categorical(df['dataset'], categories=get_datasets_order(), ordered=True)

    df = df.query('alpha == @alpha')
    x_values = get_datasets_order()
    if type in ['time', 'joint']:
        metrics = [
            'coverage',
            'relative_length',
            'geom_specialized_length',
            'wsc',
            'cond_coverage_error_z',
        ]
    else:
        metrics = [
            'coverage',
            'specialized_length',
            'wsc',
            'cond_coverage_error_z',
        ]
    conformalizers = df.conformalizer.unique()
    nrows, ncols = len(metrics), len(x_values)

    x_size = len(x_values) * len(conformalizers) ** 0.5 * 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(x_size, nrows * 1), squeeze=False)

    color_palette = generate_color_palette(conformalizer_order)

    for i, metric in enumerate(metrics):
        for j, x_value in enumerate(x_values):
            axis = axes[i, j]
            df_plot = df.query(f'{x} == @x_value')
            df_plot['dataset'] = df_plot['dataset'].astype('string')
            g = sns.barplot(
                data=df_plot,
                x=x,
                y=metric,
                hue='conformalizer',
                hue_order=conformalizer_order,
                palette=color_palette,
                ax=axis,
            )
            add_hatches(g, conformalizer_order, 1)
            if g.legend_ is not None:
                g.legend_.remove()
            axis.set_xlabel(None)
            axis.yaxis.set_tick_params(labelsize=6)
            axis.tick_params(axis='y', which='major', pad=1)

            if i != len(metrics) - 1:
                axis.set_xticks([])

            axis.axhline(0, color='black', linewidth=0.5)
            if j == 0:
                axis.set_ylabel(map_metric(metric))
            else:
                axis.set_ylabel(None)
            if metric in ['coverage', 'wsc']:
                axis.set_ylim(0, 1)
                axis.axhline(y=1 - alpha, color='black', linestyle='--')

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.9),
        frameon=True,
        ncol=7,
        fontsize=9,
    )

    if type == 'joint':
        wspace, hspace = 0.28, 0.1
    elif type == 'time':
        wspace, hspace = 0.28, 0.1
    elif type == 'mark':
        wspace, hspace = 0.38, 0.12

    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.align_labels()
    return fig


def plot_coverage(df, metric='coverage', ncols=4, type='joint'):
    df = df.copy()
    df = df.drop(columns=['metrics'])
    df = make_df_mean(df).reset_index()

    df = rename_and_filter_conformalizers(df, type)
    conformalizer_order = get_conformalizers_order(type)
    conformalizers = df.conformalizer.unique()

    df['dataset'] = pd.Categorical(df['dataset'], categories=get_datasets_order(), ordered=True)
    df = df.query('dataset in @get_datasets_order()')
    colors = mpl.colors.TABLEAU_COLORS
    assert len(conformalizers) <= len(colors)

    group = ['dataset', 'model']
    df_grouped = df.groupby(group, sort=False, observed=True)
    size = len(df_grouped)
    nrows = math.ceil(size / ncols)
    if size < ncols:
        ncols = size
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.1, nrows * 2.4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    ax = ax.flatten()
    for i in range(size, len(ax)):
        ax[i].set_visible(False)
    df['coverage_level'] = 1 - df['alpha']

    for axis, (group_values, df_model) in zip(ax, df_grouped):
        g = sns.lineplot(
            data=df_model,
            x='coverage_level',
            y=metric,
            hue='conformalizer',
            hue_order=conformalizer_order,
            ax=axis,
            err_style='bars',
            errorbar=('se', 1),
            err_kws={'capsize': 3},
            marker='o',
            markeredgewidth=0,
        )
        g.legend().remove()
        axis.plot([0, 1], [0, 1], color='black', linestyle='--')
        axis.set_xlabel(r'Coverage level $1 - \alpha$')
        axis.set_ylabel(f'Empirical {metric}')
        axis.set_aspect('equal', adjustable='box')

        dataset, model = group_values
        title = dataset
        axis.set_title(title, fontsize=12)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1),
        frameon=True,
        ncol=7,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_metrics_on_synthetic_datasets(df, alpha, type):
    # One metric per column and one dataset per row
    df = df.query('alpha == @alpha')
    df = df.query('dataset == "Hawkes"')
    df = df.query('cal_size > 0')
    df = df.reset_index()
    df = rename_and_filter_conformalizers(df, type)
    conformalizer_order = get_conformalizers_order(type)

    datasets = df.dataset.unique()
    metrics = ['coverage', 'wsc', 'relative_length']
    conformalizers = df.conformalizer.unique()
    nrows, ncols = len(datasets), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.5), squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            axis = axes[i, j]
            # for conformalizer in conformalizers:
            #     df_plot = df.query('dataset == @dataset and conformalizer == @conformalizer')
            df_plot = df.query('dataset == @dataset')
            g = sns.lineplot(
                data=df_plot,
                x='cal_size',
                y=metric,
                hue='conformalizer',
                hue_order=get_conformalizers_order_joint(),
                ax=axis,
                err_style='bars',
                errorbar=('se', 1),
                err_kws={'capsize': 3},
                marker='o',
                markeredgewidth=0,
            )
            g.legend().remove()
            axis.set_xscale('log')
            axis.set_xticks(
                list(df_plot['cal_size'].unique()),
                list(df_plot['cal_size'].unique()),
                rotation=40,
            )
            axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            axis.set_xlabel(r'Calibration size')
            axis.set_ylabel(f'{metric}')
            # axis.set_ylabel(f'{metric} on {dataset}')
            # axis.set_title(metric)
            if metric in ['coverage', 'wsc']:
                # axis.set_ylim(0, 1)
                axis.set_ylim(0.6, 1)
                axis.axhline(y=1 - alpha, color='black', linestyle='--')
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        # title='Regularization',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.95),
        frameon=True,
        ncol=5,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_metrics_for_varying_trainingsize(df, alpha, type):
    # One metric per column and one dataset per row
    df = df.query('alpha == @alpha')
    # df = df.query('dataset == @dataset')
    df = df.query('cal_size.isna()')
    df = df.reset_index()
    datasets = df.dataset.unique()
    #   print(datasets)
    metrics = ['coverage', 'wsc', 'length', 'cond_coverage_error_z']
    # metrics = ['coverage']
    conformalizers = df.conformalizer.unique()
    nrows, ncols = len(datasets), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.5), squeeze=False)
    for i, dataset in enumerate(datasets):
        df_plot = df.query('dataset == @dataset')
        for j, metric in enumerate(metrics):
            axis = axes[i, j]
            # for conformalizer in conformalizers:
            #     df_plot = df.query('dataset == @dataset and conformalizer == @conformalizer')
            if type == 'time':
                order = get_conformalizers_order_time()
            elif type == 'mark':
                order = get_conformalizers_order_mark()
            else:
                order = get_conformalizers_order_joint()
            g = sns.lineplot(
                data=df_plot,
                x='train_size',
                y=metric,
                hue='conformalizer',
                hue_order=order,
                ax=axis,
                err_style='bars',
                errorbar=('se', 1),
                err_kws={'capsize': 3},
                marker='o',
                markeredgewidth=0,
            )
            g.legend().remove()
            # axis.set_xscale('log')
            # axis.set_xticks(list(df_plot['train_size'].unique()), list(df_plot['train_size'].unique()), rotation=40)
            axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            axis.set_xlabel(r'Training set size')
            axis.set_ylabel(f'{map_metric(metric)}')
            axis.set_title(dataset)
            # axis.set_ylabel(f'{metric} on {dataset}')
            # axis.set_title(metric)
            if metric in ['coverage', 'wsc']:
                # axis.set_ylim(0, 1)
                # axis.set_ylim(0.6, 1)
                axis.axhline(y=1 - alpha, color='black', linestyle='--')
            # if metric in ['length']:
            #    axis.set_ylim(0, 250)
    axes[0,0].text(0, 0.75, dataset, va='center', ha='left', rotation='vertical', fontsize=16)
    handles, labels = axis.get_legend_handles_labels()
    # labels = np.unique(labels)
    fig.legend(
        handles,
        labels,
        # title='Regularization',
        loc='lower center',
        bbox_to_anchor=(0.5, 1),
        frameon=True,
        ncol=5,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_CCE_partition(zs, kmeans):
    fig, ax = plt.subplots(figsize=(6, 4), sharex=True, sharey=True)
    colors = sns.color_palette()
    labels = kmeans.labels_
    for z, label in zip(zs, labels):
        color = colors[label]
        ax.plot(z, np.linspace(0, 1, len(z)), color=color)
    counts = np.bincount(kmeans.labels_, minlength=kmeans.n_clusters)
    label_order = np.argsort(counts)[::-1]
    for i, label in enumerate(label_order):
        color = colors[label]
        count = np.sum(labels == label)
        ax.plot([0], [0], color=color, label=f'$|A_{{{i+1}}}| = {count}$')

    ax.set_xlabel('z')
    ax.set_ylabel('F_Z(z)')
    ax.set_xlim(1e-5, 1e4)
    ax.set_ylim(0, 1)
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig
