import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import numpy as np

mpl.rcParams['axes.formatter.use_mathtext'] = True


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


def plot_density_and_intervals(axis, intervals, dist, y_values, fill_color='red', weight=1):
    densities = dist.log_prob(y_values).exp() * weight
    for i, (left, right) in enumerate(intervals):
        mask = (left <= y_values) & (y_values <= right)
        x = y_values[mask]
        y2 = densities[mask]
        y1 = torch.zeros_like(y2)
        label = r'$1 - \alpha$' if i == 0 else None
        axis.fill_between(x=x, y1=y1, y2=y2, alpha=0.2, color=fill_color, label=label)
        label = r'$\mathrm{HDR}_\tau(1 - \alpha)$' if i == 0 else None
        axis.plot(x, y1, color='green', linewidth=7, solid_capstyle='butt', label=label)


def plot_pmf_on_side(fig, axes):
    for i in range(3):
        axis = axes[i]
        color = 'tab:red'
        if i in included:
            color = 'tab:green'
        # Get the bounds of the subplot in figure coordinates
        bounds = axis.get_position().bounds
        left, bottom, width, height = bounds

        rect_left = left + width  # x-coordinate (right side of the subplot)
        rect_width = mix_dist.probs[i] * 0.3

        height_ratio = 0.25
        rect_bottom = bottom + height * (1 - height_ratio) / 2
        rect_height = height * height_ratio

        rect = mpl.patches.Rectangle((rect_left, rect_bottom), rect_width, rect_height, transform=fig.transFigure, facecolor=color, edgecolor='black')
        fig.patches.append(rect)


def add_separator_for_marginal_time(fig, axes):
    bounds = axes[-1].get_position().bounds
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
mix_dist = D.Categorical(torch.tensor([0.5, 0.3, 0.2]))
marg_dist = Mixture(mix_dist, dists)
alpha = 0.39
included = [0, 1]


def plot():
    fig, axes = plt.subplots(4, 1, figsize=(4, 4), sharex=True, sharey=True)

    y_values = torch.linspace(1e-4, 5, 1000)
    densities = marg_dist.log_prob(y_values).exp()
    axes[-1].plot(y_values, densities, label=rf'$f(\tau)$')

    threshold, intervals = hdr_intervals(marg_dist, alpha / 2, y_values)
    axes[-1].axhline(threshold, color='black', linestyle='--', label=r'$z_{1-\alpha}$')
    plot_density_and_intervals(axes[-1], intervals, marg_dist, y_values)

    for i, (dist, weight, axis) in enumerate(zip(dists, mix_dist.probs, axes)):
        densities = dist.log_prob(y_values).exp() * weight
        axis.plot(y_values, densities, label=rf'$f(\tau, k_{i+1})$')
        if i in included:
            plot_density_and_intervals(axis, intervals, dist, y_values, fill_color='blue', weight=weight)

    axes[-1].set_xlabel(r'$\tau$', labelpad=0)
    for axis in axes:
        # Label close to the axis
        #axis.set_xlabel(r'$\tau$', labelpad=0)
        axis.set_yticks([0, 0.2, 0.4])
        axis.set_xlim(left=0, right=5)
        axis.set_ylim(bottom=0, top=0.58)
        axis.tick_params(axis='both', labelsize=9)
        # Only show the first label in the legend
        handles, labels = axis.get_legend_handles_labels()
        handles, labels = handles[:1], labels[:1]
        axis.legend(handles, labels, loc='upper right')

    # Show the other labels in the global legend
    handles, labels = axes[-1].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.99))

    plt.subplots_adjust(wspace=0, hspace=0)

    # We have to plot the pmf after plt.subplots_adjust
    plot_pmf_on_side(fig, axes)
    add_separator_for_marginal_time(fig, axes)
    add_separator_for_marginal_mark(fig, axes)
    for interval in intervals:
        add_vertical_dotted_line(fig, axes, interval[0])
        add_vertical_dotted_line(fig, axes, interval[1])

    savefig('joint_individual.pdf')


plot()
