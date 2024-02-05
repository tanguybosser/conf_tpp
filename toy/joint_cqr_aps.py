import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import numpy as np

mpl.rcParams['axes.formatter.use_mathtext'] = True

plot_densities = True


class Mixture:
    def __init__(self, mix, comps):
        self.mix = mix
        self.comps = comps
    
    def log_prob(self, x):
        log_probs = [comp.log_prob(x) for comp in self.comps]
        log_probs = torch.stack(log_probs, dim=-1)
        weighted_log_probs = self.mix.logits + log_probs
        return weighted_log_probs.logsumexp(-1)
    
    def cdf(self, x):
        cdf_values = [comp.cdf(x) for comp in self.comps]
        cdf_values = torch.stack(cdf_values, dim=-1)
        return (self.mix.probs * cdf_values).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        mix_sample = self.mix.sample(sample_shape)
        samples = [self.comps[index].sample().item() for index in mix_sample.flatten()]
        return torch.tensor(samples, dtype=torch.float).reshape(mix_sample.shape)


def create_interval(seq):
    # Return the intervals of True values in the sequence
    seq = torch.cat([torch.tensor([False], device=seq.device), seq, torch.tensor([False], device=seq.device)])
    diffs = seq[1:].int() - seq[:-1].int()
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze(-1)
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze(-1) - 1
    return torch.stack((starts, ends + 1), dim=1)


def hdr_intervals(dist, alpha, y_values, n_sample=100000):
    y_sample = dist.sample(torch.Size([n_sample]))
    z_sample = dist.log_prob(y_sample).exp()
    # Quantile of F_Z
    z_quantile = torch.quantile(z_sample, alpha)
    threshold = z_quantile

    densities = dist.log_prob(y_values).exp()
    valid = densities > threshold
    index_intervals = create_interval(valid)
    index_intervals = torch.minimum(index_intervals, torch.tensor(y_values.shape[0] - 1))
    intervals = y_values[index_intervals]
    return threshold, intervals


def plot_intervals_and_coverage(axis, intervals, dist, y_values, fill_type, weight=1, plot_densities=True):
    if fill_type == 'marginal':
        fill_color = 'red'
        fill_label = r'$1 - \frac{\alpha}{2}$'
    elif fill_type == 'joint':
        fill_color = 'blue'
        fill_label = r'$\geq 1 - \alpha$'
    densities = dist.log_prob(y_values).exp() * weight
    for i, (left, right) in enumerate(intervals):
        mask = (left <= y_values) & (y_values <= right)
        x = y_values[mask]
        y2 = densities[mask]
        y1 = torch.zeros_like(y2)
        label = r'$[Q_\tau(\frac{\alpha}{4}), Q_\tau(1 - \frac{\alpha}{4})]$' if i == 0 else None
        axis.plot(x, y1, color='green', linewidth=7, solid_capstyle='butt', label=label)
        label = fill_label if i == 0 else None
        if plot_densities:
            axis.fill_between(x=x, y1=y1, y2=y2, alpha=0.2, color=fill_color, label=label)


def plot_pmf_on_side(fig, axes, axis_right):
    for i in range(3):
        axis = axes[i]
        color = 'tab:red'
        alpha = 1
        if i in included:
            alpha = 0.2
        # Get the bounds of the subplot in figure coordinates
        bounds = axis.get_position().bounds
        left, bottom, width, height = bounds

        _, _, right_width, _ = axis_right.get_position().bounds

        rect_left = left + width  # x-coordinate (right side of the subplot)
        rect_width = mix_dist.probs[i] * 0.19

        height_ratio = 0.25
        rect_bottom = bottom + height * (1 - height_ratio) / 2
        rect_height = height * height_ratio

        edgecolor = 'red' if i in included else 'black'
        rect = mpl.patches.Rectangle((rect_left + right_width - rect_width, rect_bottom), rect_width, rect_height, transform=fig.transFigure, facecolor=color, alpha=alpha, edgecolor=edgecolor)
        fig.patches.append(rect)


def add_separator_for_marginal_time(fig, axis_bottom):
    bounds = axis_bottom.get_position().bounds
    left, bottom, width, height = bounds
    rect_height = 0.005
    rect = mpl.patches.Rectangle((left, bottom + height - rect_height), width, rect_height, transform=fig.transFigure, facecolor='black', edgecolor='black')
    fig.patches.append(rect)


def add_separator_for_marginal_mark(fig, axes):
    bounds = axes[2].get_position().bounds
    left, bottom, width, height = bounds
    rect_width = 0.005
    rect = mpl.patches.Rectangle((left + width - rect_width, bottom), rect_width, height * 3, transform=fig.transFigure, facecolor='black', edgecolor='black')
    fig.patches.append(rect)


def add_vertical_dotted_line(fig, axes, x):
    con = ConnectionPatch(
        xyA=(x, 0),
        xyB=(x, 0),
        coordsA=axes[0].get_xaxis_transform(),
        coordsB=axes[3].get_xaxis_transform(),
        axesA=axes[0],
        axesB=axes[3],
        arrowstyle='-', 
        linestyle='dotted', 
        color='tab:green', 
        lw=1
    )
    fig.add_artist(con)


def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends + fig.patches or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)


dists = [
    D.LogNormal(0.5, 0.5),
    D.LogNormal(0.5, 1.0),
    D.MixtureSameFamily(D.Categorical(torch.tensor([0.5, 0.5])), D.LogNormal(torch.tensor([0., 1.]), torch.tensor([0.2, 0.05]))),
]
mix_dist = D.Categorical(torch.tensor([0.55, 0.25, 0.2]))
marg_dist = Mixture(mix_dist, dists)
alpha = 0.39
included = [0, 1]


def plot():
    #fig, axes = plt.subplots(4, 1, figsize=(4, 4), sharex=True, sharey=True)

    fig = plt.figure(figsize=(6.4, 4.8))
    gs = GridSpec(4, 4)
    axis_bottom = fig.add_subplot(gs[-1,0:3])
    axes = np.array([
        fig.add_subplot(gs[i, 0:3], sharex=axis_bottom, sharey=axis_bottom)
        for i in range(3)
    ])
    axes_vertical = np.array([*axes, axis_bottom])
    axis_right = fig.add_subplot(gs[0:3, -1])

    y_values = torch.linspace(1e-4, 5, 1000)
    densities = marg_dist.log_prob(y_values).exp()
    axis_bottom.plot(y_values, densities, color='black')#, label=rf'$f(\tau)$')

    #threshold, intervals = hdr_intervals(marg_dist, alpha / 2, y_values)
    intervals = [(0.55, 3)]
    plot_intervals_and_coverage(axis_bottom, intervals, marg_dist, y_values, fill_type='marginal', plot_densities=True)

    for i, (dist, weight, axis) in enumerate(zip(dists, mix_dist.probs, axes)):
        densities = dist.log_prob(y_values).exp() * weight
        if plot_densities:
            axis.plot(y_values, densities, label=rf'$f(\tau, k_{i+1})$')
        if i in included:
            plot_intervals_and_coverage(axis, intervals, dist, y_values, fill_type='joint', weight=weight, plot_densities=plot_densities)

    for axis in axes_vertical:
        axis.set_yticks([0, 0.2, 0.4])
        axis.set_xlim(left=0, right=5)
        axis.set_ylim(bottom=0, top=0.58)
        axis.tick_params(axis='both', labelsize=9)
    
    for i, axis in enumerate(axes):
        axis.tick_params(labelbottom=False, length=0)
        handles, labels = axis.get_legend_handles_labels()
        if plot_densities:
            if i == 0:
                handles = [handles[0], handles[2]]
                labels = [labels[0], labels[2]]
            else:
                handles, labels = handles[:1], labels[:1]
            axis.legend(handles, labels, loc='upper right', fontsize=8.4)

    axis_bottom.set_xlabel(r'$\tau$', labelpad=0)
    axis_bottom.set_ylabel(rf'$f(\tau)$', labelpad=0)
    handles, labels = axis_bottom.get_legend_handles_labels()
    handles, labels = handles[:2], labels[:2]
    axis_bottom.legend(handles, labels, loc='upper right', fontsize=8.4)

    axis_right.set_xlabel(r'$f(k)$', labelpad=0)
    axis_right.set_yticks([])
    axis_right.invert_xaxis()
    axis_right.set_xticks([0.5, 0.])
    axis_right.legend(handles[1:], labels[1:], loc='lower left', fontsize=8.4)

    # # Show the other labels in the global legend
    # handles, labels = axes[0].get_legend_handles_labels()
    # handles, labels = handles[1:], labels[1:]
    # fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(wspace=0, hspace=0)

    # We have to plot the pmf after plt.subplots_adjust
    plot_pmf_on_side(fig, axes, axis_right)
    add_separator_for_marginal_time(fig, axis_bottom)
    add_separator_for_marginal_mark(fig, axes)
    for interval in intervals:
        add_vertical_dotted_line(fig, axes_vertical, interval[0])
        add_vertical_dotted_line(fig, axes_vertical, interval[1])

    savefig('joint_cqr_aps.pdf')


plot()
