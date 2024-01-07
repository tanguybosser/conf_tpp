from collections import defaultdict
import json

import torch
import numpy as np

from .metrics import (
    compute_regions_coverage, 
    compute_regions_length,
    get_repr_samples_from_dl, 
    wsc_unbiased, 
    ConditionalCoverageComputerForRepr,
    ConditionalCoverageComputerForZ,
)
from .tpp_util import at_last_event, until_last_event, get_history_and_target
from .conformalizer import ConformalizerBase


def debug(cond_cov_z, cond_coverages, dl_calib):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(np.bincount(cond_cov_z.kmeans.labels_, minlength=cond_cov_z.kmeans.n_clusters))
    #array([ 1, 33,  8,  1,  3, 54,  4, 21,  2,  1])
    print(cond_coverages)
    #tensor([1.0000, 0.7917, 0.4000, 0.5000, 0.5000, 0.9444, 1.0000, 0.8571, 0.6667, 0.5000])
    from .metrics import get_logz_samples_from_dl
    logzs = get_logz_samples_from_dl(cond_cov_z.model, dl_calib, cond_cov_z.args)
    zs = logzs.exp()

    fig, ax = plt.subplots(figsize=(10, 5), sharex=True, sharey=True)
    colors = sns.color_palette()
    labels = cond_cov_z.kmeans.labels_
    for z, label in zip(zs, labels):
        ax.plot(z, np.linspace(0, 1, len(z)), drawstyle='steps-pre', color=colors[label])
    ax.set_xscale('log')
    plt.show()


def plot_z(zs, kmeans):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    plt.savefig('test.pdf')


class Tester:
    def __init__(self, dl_test, dl_calib, model, alpha, conformalizer: ConformalizerBase, rc, args):
        self.dl_test = dl_test
        self.dl_calib = dl_calib
        self.model = model
        self.alpha = alpha
        self.conformalizer = conformalizer
        self.rc = rc
        self.args = args
    
    def compute_test_metrics(self):
        metrics_to_average = defaultdict(list)
        # We can't use torch.no_grad() else there is an error when computing the CDF of the tpps package
        #with torch.no_grad():
        preds = []
        for batch in self.dl_test:
            metrics_on_batch = self.compute_on_batch(batch)
            for name, values in metrics_on_batch.items():
                if name in ['coverage', 'joint_length', 'specialized_length']:
                    metrics_to_average[name].append(values)
            preds.extend(metrics_on_batch['preds'])
        metrics_to_average = {
            name: torch.concat(values).float()
            for name, values in metrics_to_average.items()
        }
        metrics = {
            name: values.mean().cpu().item()
            for name, values in metrics_to_average.items()
        }
        metrics['preds'] = preds
        metrics['type'] = self.conformalizer.get_predictor_type()

        if self.args.eval_cond_coverage:
            if 'hawkes' not in self.args.model_name_short:
                metrics['wsc'] = self.compute_wsc(metrics_to_average['coverage']).item()

                cond_cov_repr = ConditionalCoverageComputerForRepr(self.model, self.alpha, self.args)
                cond_cov_repr.compute_partition(self.dl_calib, nb_partitions=self.args.n_partitions)
                cond_coverages = cond_cov_repr.compute_cond_coverages(metrics_to_average['coverage'], self.dl_test)
                error = cond_cov_repr.compute_error(cond_coverages).item()
                metrics['cond_coverages_repr'] = cond_coverages
                metrics['cond_coverage_error_repr'] = error

            cond_cov_z = ConditionalCoverageComputerForZ(self.model, self.alpha, self.args)
            cond_cov_z.compute_partition(self.dl_calib, nb_partitions=self.args.n_partitions)
            cond_coverages = cond_cov_z.compute_cond_coverages(metrics_to_average['coverage'], self.dl_test)
            error = cond_cov_z.compute_error(cond_coverages).item()
            metrics['cond_coverages_z'] = cond_coverages
            metrics['cond_coverage_error_z'] = error
            print(cond_coverages)
            print(np.bincount(cond_cov_z.kmeans.labels_, minlength=cond_cov_z.kmeans.n_clusters))

        

        return metrics

    def compute_on_batch(self, batch):
        past_events, target_time, target_label = get_history_and_target(batch, self.args)
        preds = self.conformalizer.get_joint_prediction_region(past_events)

        coverage = compute_regions_coverage(preds, target_time, target_label)
        joint_length = compute_regions_length(preds, type='joint')
        predictor_type = self.conformalizer.get_predictor_type()
        specialized_length = compute_regions_length(preds, type=predictor_type)

        return {
            'coverage': coverage,
            'joint_length': joint_length,
            'specialized_length': specialized_length,
            'preds': preds,
        }

    # Some metrics have to be computed on the whole test set and not on individual batches
    def compute_wsc(self, coverages):
        reprs = get_repr_samples_from_dl(self.model, self.dl_test, self.args)
        return wsc_unbiased(reprs.numpy(), coverages.numpy(), delta=0.2)
