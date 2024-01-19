import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['axes.formatter.use_mathtext'] = True


from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np

from tueplots import axes


class Mixture:
    # Custom Mixture class to handles mixture of distributions of different classes
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
        comp_sample = torch.tensor(samples, dtype=torch.float).reshape(mix_sample.shape)
        return mix_sample, comp_sample


def create_interval(seq):
    # Return the intervals of True values in the sequence
    seq = torch.cat([torch.tensor([False], device=seq.device), seq, torch.tensor([False], device=seq.device)])
    diffs = seq[1:].int() - seq[:-1].int()
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze(-1)
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze(-1) - 1
    return torch.stack((starts, ends + 1), dim=1)


def plot_density_and_intervals(axis, intervals, dist, y_values, fill_color='red', weight=1):
    densities = dist.log_prob(y_values).exp() * weight
    for i, (left, right) in enumerate(intervals):
        mask = (left <= y_values) & (y_values <= right)
        x = y_values[mask]
        y2 = densities[mask]
        y1 = torch.zeros_like(y2)
        label = r'$\mathrm{HDR}_{\tau, k}(1 - \alpha)$' if i == 0 else None
        axis.plot(x, y1, color='green', linewidth=7, solid_capstyle='butt', label=label)
        label = r'$1 - \alpha$' if i == 0 else None
        axis.fill_between(x=x, y1=y1, y2=y2, alpha=0.2, color=fill_color, label=label)


def hdr_threshold(dists, mix_dist, alpha, n_sample=10000):
    marg_dist = Mixture(mix_dist, dists)
    mix_samples, comp_samples = marg_dist.sample(torch.Size([n_sample]))
    z_sample = torch.tensor([
        dists[mix_sample].log_prob(comp_sample).exp() * mix_dist.probs[mix_sample]
        for mix_sample, comp_sample in zip(mix_samples, comp_samples)
    ])
    z_quantile = torch.quantile(z_sample, alpha)
    threshold = z_quantile
    return threshold


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


dists = [
    D.LogNormal(0.5, 0.5),
    D.LogNormal(0.5, 1.0),
    D.MixtureSameFamily(D.Categorical(torch.tensor([0.5, 0.5])), D.LogNormal(torch.tensor([0., 1.]), torch.tensor([0.2, 0.05]))),
]
mix_dist = D.Categorical(torch.tensor([0.55, 0.25, 0.2]))

alpha = 0.4
threshold = hdr_threshold(dists, mix_dist, alpha=alpha)


def plot():
    fig, axes = plt.subplots(3, 1, figsize=(4.7, 3.6), sharex=True, sharey=True)
    y_values = torch.linspace(1e-4, 5, 1000)

    for i, (dist, weight, axis) in enumerate(zip(dists, mix_dist.probs, axes)):
        densities = dist.log_prob(y_values).exp() * weight
        axis.plot(y_values, densities, label=rf'$f(\tau, k_{i+1})$', color='black')
        
        axis.axhline(threshold, color='black', linestyle='--', label=r'$z_{1-\alpha}$')

        valid = densities > threshold
        index_intervals = create_interval(valid)
        index_intervals = torch.minimum(index_intervals, torch.tensor(y_values.shape[0] - 1))
        intervals = y_values[index_intervals]
        print(intervals)

        plot_density_and_intervals(axis, intervals, dist, y_values, weight=weight)
        

    axes[-1].set_xlabel(r'$\tau$', labelpad=0)
    for axis in axes:
        # Label close to the axis
        axis.set_xlim(left=0, right=5)
        axis.set_yticks([0, 0.2, 0.4])
        axis.set_ylim(bottom=0, top=0.58)
        axis.tick_params(axis='both', labelsize=9)
        # Only show the first label in the legend
        handles, labels = axis.get_legend_handles_labels()
        handles, labels = handles[:1], labels[:1]
        axis.legend(handles, labels, loc='upper right')

    # Show the other labels in the global legend
    handles, labels = axes[-1].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

    plt.subplots_adjust(wspace=0, hspace=0)
    savefig('joint_hdr.pdf')


plot()
