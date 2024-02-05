import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

mpl.rcParams['axes.formatter.use_mathtext'] = True

K = 20

pmfs = torch.randn((3, K)).abs()
pmfs[0] *= 1.5
pmfs[2] *= 0.5
pmfs = pmfs.softmax(dim=-1)
sorted = pmfs.sort(dim=-1, descending=True)
order = sorted.indices
pmfs = sorted.values

y_indices = torch.tensor([0, 12, 17])

alpha = 0.5

n = len(pmfs)


class ConformalizerBase:
    def __init__(self, proba, target_label_index):
        self.device = 'cpu'
        self.alpha = alpha
        self.scores = self.get_score(proba, target_label_index)
        self.q = self.conformal_quantile(self.scores)

    def get_score(self, inter_time_dist, inter_times, seq_len):
        pass

    def conformal_quantile(self, scores):
        n = scores.shape[0]
        level = torch.ceil(torch.tensor((1 - self.alpha) * (n + 1))) / (n + 1)
        if level > 1:
            return torch.tensor(self.max_score(), device=scores.device)
        level = level.type(scores.dtype).to(scores.device)
        return torch.quantile(scores, level)
    
    def get_mark_prediction_region(self, proba):
        pass

class APS(ConformalizerBase):
    def get_score(self, proba, target_label_index):
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        score_idx = torch.where(pi == target_label_index.unsqueeze(-1))[1]
        rand = torch.rand(n).to(self.device) * srt[range(n), score_idx]
        score = torch.cumsum(srt, dim=1)[range(n), score_idx] - rand 
        return score
    
    def get_mark_prediction_region(self, proba):
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        srt_cumsum = torch.cumsum(srt, dim=1)
        rand = torch.rand(n,1).to(self.device) * srt
        ind = srt_cumsum - rand <= self.q 
        ind[:,0] = True #Avoids 0 size sets
        prediction_sets = torch.take_along_dim(ind, pi.argsort(dim=1), dim=1)
        return prediction_sets

class NaiveAPS(APS):
    def get_score(self, proba, target_label):
        batch_shape = proba.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.device)


class RAPS(APS):    
    def get_score(self, proba, target_label_index, lambda_reg=0.01, k_reg=3):
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        reg = torch.tensor(k_reg*[0,] + (proba.shape[1]-k_reg)*[lambda_reg,])[None,:]
        reg = reg.to(self.device)
        srt_reg = srt + reg
        score_idx = torch.where(pi == target_label_index.unsqueeze(-1))[1]
        rand = torch.rand(n).to(self.device) * srt_reg[range(n), score_idx]
        score = torch.cumsum(srt_reg, dim=1)[range(n), score_idx] - rand 
        return score
    
    def get_mark_prediction_region(self, proba, lambda_reg=0.01, k_reg=3):
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        reg = torch.tensor(k_reg*[0,] + (proba.shape[1]-k_reg)*[lambda_reg,])[None,:]
        reg = reg.to(self.device)
        srt_reg = srt + reg
        srt_reg_cumsum = torch.cumsum(srt_reg, dim=1)
        rand = torch.rand(n,1).to(self.device) * srt_reg
        ind = srt_reg_cumsum - rand <= self.q
        ind[:,0] = True #Avoids 0 size sets
        prediction_sets = torch.take_along_dim(ind, pi.argsort(dim=1), dim=1)
        return prediction_sets


class NaiveRAPS(RAPS):
    def get_score(self, proba, target_label_index):
        batch_shape = proba.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.device)


class ConformalizedProb(ConformalizerBase):
    def get_score(self, proba, target_label_index):
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        score = 1 - proba[range(proba.shape[0]), target_label_index]
        return score

    def get_mark_prediction_region(self, proba):
        prediction_sets = proba >= (1 - self.q) #I think that there is no guarantee against empty prediction sets. 
        return prediction_sets


def plot_selected_marks(axis, region, index, **kwargs):
    left = 0
    true_indices = torch.where(region)[0]
    last_true_index = true_indices[-1] if len(true_indices) > 0 else -1
    right = last_true_index
    axis.add_patch(plt.Rectangle((left - 0.5, index - 0.1), right - left + 1, 0.2, fill=True, edgecolor='white', **kwargs))


def plot():
    fig, axes = plt.subplots(2, n, figsize=(8, 2.5), sharex=True, sharey='row')
    x_values = torch.arange(K)

    haps_region = NaiveAPS(pmfs, y_indices).get_mark_prediction_region(pmfs)
    hraps_region = NaiveRAPS(pmfs, y_indices).get_mark_prediction_region(pmfs)
    cprob_region = ConformalizedProb(pmfs, y_indices).get_mark_prediction_region(pmfs)
    caps_region = APS(pmfs, y_indices).get_mark_prediction_region(pmfs)
    craps_region = RAPS(pmfs, y_indices).get_mark_prediction_region(pmfs)

    for i, y_index in enumerate(y_indices):
        pmf = pmfs[i]

        axes[0, i].bar(x_values, pmf, label=rf'$p(k | \mathbf{{h}}_{i + 1})$', color='white', edgecolor='black')
        axes[0, i].set_ylim(bottom=0)
        axes[0, i].set_xlim((0 - 0.8, K - 1 + 0.8))

        axes[0, i].axvline(y_index, color='black', linestyle='--')#, label=r'$y$')
        axes[1, i].axvline(y_index, color='black', linestyle='--')#, label=r'$y$')

        ids = np.arange(0, -1, -0.2)

        plot_selected_marks(axes[1, i], haps_region[i], index=ids[0], facecolor='tab:blue', hatch='//')
        plot_selected_marks(axes[1, i], hraps_region[i], index=ids[1], facecolor='tab:orange', hatch='//')
        plot_selected_marks(axes[1, i], cprob_region[i], index=ids[2], facecolor='tab:green')
        plot_selected_marks(axes[1, i], caps_region[i], index=ids[3], facecolor='tab:blue')
        plot_selected_marks(axes[1, i], craps_region[i], index=ids[4], facecolor='tab:orange')

        if i == 0:
            axes[1, i].set_yticks(ids)
        custom_labels = ['H-APS', 'H-RAPS', 'C-PROB', 'C-APS', 'C-RAPS']
        axes[1, i].set_yticklabels(custom_labels, fontsize=9)
        axes[1, i].set_ylim(-1., 0.22)

        axes[1, i].set_xticks(x_values)
        axes[1, i].set_xticklabels(order[i].numpy(), fontsize=5)

        axes[0, i].legend(loc='upper right', fontsize=9)


    for axis in axes[-1]:
        # Label close to the axis
        axis.set_xlabel(r'$k$', labelpad=3)

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
savefig('example_regions_for_k.pdf')
