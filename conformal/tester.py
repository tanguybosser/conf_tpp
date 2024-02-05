import json
from collections import defaultdict

import numpy as np
import torch

from .conformalizer import ConformalizerBase
from .metrics import (
    ConditionalCoverageComputerForRepr,
    ConditionalCoverageComputerForZ,
    compute_regions_coverage,
    compute_regions_length,
    get_reprs_from_dl,
    gmean,
    wsc_unbiased,
)
from .tpp_util import get_history_and_target


class Tester:
    def __init__(
        self,
        dl_test,
        dl_calib,
        model,
        alpha,
        conformalizer: ConformalizerBase,
        rc,
        args,
    ):
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
        # with torch.no_grad():
        preds = []
        for batch in self.dl_test:
            metrics_on_batch = self.compute_on_batch(batch)
            for name, values in metrics_on_batch.items():
                if name in ['coverage', 'joint_length', 'specialized_length']:
                    metrics_to_average[name].append(values)
            preds.extend(metrics_on_batch['preds'])
        metrics_to_average = {
            name: torch.cat(values).float().cpu() for name, values in metrics_to_average.items()
        }
        metrics = {name: values.mean().item() for name, values in metrics_to_average.items()}
        metrics['geom_specialized_length'] = gmean(metrics_to_average['specialized_length']).item()
        metrics['geom_joint_length'] = gmean(metrics_to_average['joint_length']).item()
        metrics['preds'] = preds
        metrics['type'] = self.conformalizer.get_predictor_type()

        if self.args.eval_cond_coverage:
            if 'hawkes' not in self.args.model_name_short:
                metrics['wsc'] = self.compute_wsc(metrics_to_average['coverage']).item()

                cond_cov_repr = ConditionalCoverageComputerForRepr(self.model, self.alpha, self.args)
                cond_cov_repr.compute_partition(self.dl_calib, nb_partitions=self.args.n_partitions)
                cond_coverages = cond_cov_repr.compute_cond_coverages(
                    metrics_to_average['coverage'], self.dl_test
                )
                error = cond_cov_repr.compute_error(cond_coverages).item()
                metrics['cond_coverages_repr'] = cond_coverages
                metrics['cond_coverage_error_repr'] = error

            cond_cov_z = ConditionalCoverageComputerForZ(self.model, self.alpha, self.args)
            cond_cov_z.compute_partition(self.dl_calib, nb_partitions=self.args.n_partitions)
            cond_coverages = cond_cov_z.compute_cond_coverages(metrics_to_average['coverage'], self.dl_test)
            error = cond_cov_z.compute_error(cond_coverages).item()
            metrics['cond_coverages_z'] = cond_coverages
            metrics['cond_coverage_error_z'] = error
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
        reprs = get_reprs_from_dl(self.model, self.dl_test, self.args)
        return wsc_unbiased(reprs.numpy(), coverages.numpy(), delta=0.2)
