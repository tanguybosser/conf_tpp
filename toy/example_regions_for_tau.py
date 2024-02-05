import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['axes.formatter.use_mathtext'] = True

# dists = [
#     D.LogNormal(-0.5, 0.5),
#     D.LogNormal(-0.5, 0.5),
#     D.LogNormal(0, 1),
#     D.LogNormal(0, 1),
#     D.LogNormal(0, 1)
# ]
# ys = [
#     0.8,
#     0.5,
#     3,
#     4,
#     4,
# ]

# dists = [
#     D.Exponential(2),
#     D.Exponential(1.4),
#     D.Exponential(0.62),
# ]
# y_alphas = [
#     0.6,
#     0.92,
#     0.93,
# ]



dists = [
    D.LogNormal(0.2, 0.5),
    D.LogNormal(0.5, 1.0),
    D.LogNormal(0.5, 1.5),
]
y_alphas = [0.95, 0.6, 0.88]


# dists = [
#     D.Exponential(rate)
#     for rate in np.linspace(2.5, 0.6, 6)
# ]
# y_alphas = [0.76, 0.95, 0.92, 0.73, 0.83, 0.75]
#y_alphas =   [0.9, 0.6, 0.7, 0.9, 0.85, 0.83]
alpha = 0.5

n = len(dists)


def plot_intervals(axis, intervals, index, **kwargs):
    for left, right in intervals:
        axis.add_patch(plt.Rectangle((left, index - 0.1), right - left, 0.2, fill=True, edgecolor='white', **kwargs))


def create_interval(seq):
    # Return the intervals of True values in the sequence
    seq = torch.cat([torch.tensor([False], device=seq.device), seq, torch.tensor([False], device=seq.device)])
    diffs = seq[1:].int() - seq[:-1].int()
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze(-1)
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze(-1) - 1
    return torch.stack((starts, ends + 1), dim=1)


def plot_hdr(axis, dist, alpha, y_values, **kwargs):
    y_sample = dist.sample(torch.Size([100000]))
    z_sample = dist.log_prob(y_sample).exp()
    # Quantile of F_Z
    z_quantile = torch.quantile(z_sample, alpha)
    threshold = z_quantile

    densities = dist.log_prob(y_values).exp()
    valid = densities > threshold
    index_intervals = create_interval(valid)
    index_intervals = torch.minimum(index_intervals, torch.tensor(y_values.shape[0] - 1))
    intervals = y_values[index_intervals]
    plot_intervals(axis, intervals, **kwargs)


def plot_cqrl(axis, dist, alpha, q=0, **kwargs):
    right = dist.icdf(torch.tensor(1 - alpha))
    plot_intervals(axis, [(0, right + q)], **kwargs)


def plot_cqr(axis, dist, alpha, q=0, **kwargs):
    left = dist.icdf(torch.tensor(alpha / 2))
    right = dist.icdf(torch.tensor(1 - alpha / 2))
    plot_intervals(axis, [(torch.maximum(left - q, torch.tensor(0.)), right + q)], **kwargs)


def plot_const(axis, q, **kwargs):
    plot_intervals(axis, [(0, q)], **kwargs)


def plot():
    fig, axes = plt.subplots(2, n, figsize=(8, 2.5), sharex=True, sharey='row')
    x_max = 12
    x_values = torch.linspace(1e-4, x_max, 1000)

    ts = {
        'const': 0,
        'cqr': 0,
        'cqrl': 0,
    }

    for i, (dist, y_alpha) in enumerate(zip(dists, y_alphas)):
        densities = dist.log_prob(x_values).exp()
        densities = densities.numpy()

        axes[0, i].plot(x_values, densities, label=rf'$f(\tau | \mathbf{{h}}_{i + 1})$', color='black')
        axes[0, i].set_ylim(bottom=0)
        axes[0, i].set_xlim((0, x_max))

        y = dist.icdf(torch.tensor(y_alpha))
        axes[0, i].axvline(y, color='black', linestyle='--') #, label=r'$y$')
        axes[1, i].axvline(y, color='black', linestyle='--') #, label=r'$y$')

        ids = np.arange(0, -1.4, -0.2)
        #value = y_alphas[quantile]
        plot_cqr(axes[1, i], dist, alpha, index=ids[0], facecolor='tab:blue', hatch='//')
        plot_cqrl(axes[1, i], dist, alpha, index=ids[1], facecolor='tab:orange', hatch='//')
        plot_hdr(axes[1, i], dist, alpha, x_values, index=ids[2], facecolor='tab:green', hatch='//')

        t = ts['const']
        q = dists[t].icdf(torch.tensor(y_alphas[t]))
        plot_const(axes[1, i], q, index=ids[3], facecolor='tab:red')

        t = ts['cqr']
        q = torch.max(
            dists[t].icdf(torch.tensor(y_alphas[t])) - dists[t].icdf(torch.tensor(1 - alpha / 2)),
            dists[t].icdf(torch.tensor(alpha / 2)) - dists[t].icdf(torch.tensor(y_alphas[t]))
        )
        print(q)
        plot_cqr(axes[1, i], dist, alpha, q=q, index=ids[4], facecolor='tab:blue')

        t = ts['cqrl']
        q = dists[t].icdf(torch.tensor(y_alphas[t])) - dists[t].icdf(torch.tensor(1 - alpha))
        plot_cqrl(axes[1, i], dist, alpha, q=q, index=ids[5], facecolor='tab:orange')

        plot_hdr(axes[1, i], dist, 0.1181, x_values, index=ids[6], facecolor='tab:green')

        axes[1, i].set_yticks(ids)
        custom_labels = ['H-QR', 'H-QRL', 'H-HDR', 'C-Const', 'C-QR', 'C-QRL', 'C-HDR']
        axes[1, i].set_yticklabels(custom_labels, fontsize=9)
        axes[1, i].set_ylim(-1.37, 0.22)
        axes[0, i].legend(loc='upper right', fontsize=9)


    for axis in axes[-1]:
        # Label close to the axis
        axis.set_xlabel(r'$\tau$', labelpad=0)

    # No space between subplots
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)



def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)

plot()
savefig('example_regions_for_tau.pdf')
