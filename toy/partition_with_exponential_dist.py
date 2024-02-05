import torch
import torch.distributions as D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

mpl.rcParams['axes.formatter.use_mathtext'] = True

dists = D.Exponential(torch.arange(0.5, 2., 0.01))

def get_z_samples():
    y_sample = dists.sample(torch.Size([1000]))
    z_sample = dists.log_prob(y_sample).exp()
    z_sample = z_sample.permute(1, 0)
    z_sample = z_sample.sort(dim=1).values
    return z_sample


class ConditionalCoverageComputer:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def get_partition_features(self, dl_calib):
        pass

    def compute_partition(self, nb_partitions):
        features = self.get_partition_features()
        self.kmeans = KMeans(n_clusters=nb_partitions).fit(features)
    
    def compute_cond_coverages(self, coverages, dl_test):
        features = self.get_partition_features(dl_test)
        test_partitions = self.kmeans.predict(features)
        cond_coverage_list = []
        for i in range(self.kmeans.n_clusters):
            cond_coverages = coverages[test_partitions == i]
            if len(cond_coverages) == 0: # Special case: no sample in the partition
                cond_coverage = 0.5
            else:
                cond_coverage = cond_coverages.mean()
            cond_coverage_list.append(cond_coverage)
        return torch.tensor(cond_coverage_list)

    def compute_error(self, cond_coverages):
        labels = self.kmeans.labels_
        weights = np.bincount(labels, minlength=self.kmeans.n_clusters) / len(labels)
        error = ((cond_coverages - (1 - self.alpha)).square() * weights).sum()
        return error


class ConditionalCoverageComputerForZ(ConditionalCoverageComputer):
    def get_partition_features(self):
        return get_z_samples()


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


def plot():
    cond_cov_z = ConditionalCoverageComputerForZ(alpha=0.2)
    cond_cov_z.compute_partition(nb_partitions=10)
    labels = cond_cov_z.kmeans.labels_


    fig, ax = plt.subplots(figsize=(8, 5), sharex=True, sharey=True)
    x_values = torch.linspace(1e-4, 4, 1000)
    dists_densities = dists.log_prob(x_values[:, None]).exp()
    dists_densities = dists_densities.permute(1, 0)
    print(np.bincount(labels, minlength=cond_cov_z.kmeans.n_clusters))
    colors = sns.color_palette()
    for dist_densities, label in zip(dists_densities, labels):
        ax.plot(x_values, dist_densities, color=colors[label])
    savefig('partition_with_exponential_dist.pdf')

plot()
