import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from .dist_util import get_logz_samples
from .tpp_util import get_history_and_target


def compute_region_coverage(pred, time, label_index):
    if label_index in pred:
        for interval in pred[label_index]:
            left, right = interval
            if left < time < right:
                return True
    return False

def compute_regions_coverage(preds, times, labels):
    coverages = []
    for pred, time, label in zip(preds, times, labels):
        label_index = label.argmax(-1).item()
        coverages.append(compute_region_coverage(pred, time, label_index))
    return torch.tensor(coverages)

def compute_region_length_joint(pred):
    length = 0
    for mark_region in pred.values():
        for interval in mark_region:
            left, right = interval
            assert left <= right
            length += right - left
    return length

def compute_region_length_time(pred):
    pred_time = pred[0]
    # We assume that the prediction is the same for all marks
    for p in pred.values():
        assert torch.tensor(p == pred_time).all(), (p, pred_time)
    length = 0
    for interval in pred_time:
        left, right = interval
        assert left <= right
        length += right - left
    return length

def compute_region_length_mark(pred):
    length = len(list(pred.keys()))
    return length 
    
def compute_regions_length(preds, type='joint'):
    assert type in ['joint', 'time', 'mark']
    lengths = []
    for pred in preds:
        if type == 'joint':
            lengths.append(compute_region_length_joint(pred))
        elif type == 'time':
            lengths.append(compute_region_length_time(pred))
        else:
            lengths.append(compute_region_length_mark(pred))
    return torch.tensor(lengths)



# wsc is adapted from https://github.com/msesia/chr/blob/master/chr/coverage.py

def wsc(reprs, coverages, delta, M=1000):
    def wsc_v(reprs, cover, delta, v):
        n = len(cover)
        z = np.dot(reprs, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best, bi_best = 0, n - 1
        cover_min = cover.mean()
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best, bi_best = ai, bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    def sample_from_normal_approximation(n, X):
        mean = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False)
        return np.random.multivariate_normal(mean, covariance, size=n)

    V = sample_sphere(M, reprs.shape[1])
    #V = sample_from_normal_approximation(M, reprs)
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    for m in range(M):
        wsc_list[m], a_list[m], b_list[m] = wsc_v(reprs, coverages, delta, V[m])
    
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased(reprs, coverages, delta, M=1000, test_size=0.75, random_state=0):
    def wsc_vab(reprs, cover, v, a, b):
        n = len(reprs)
        z = np.dot(reprs, v)
        idx = np.where((a <= z) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    reprs_train, reprs_test, coverages_train, coverages_test = train_test_split(
        reprs, coverages, test_size=test_size, random_state=random_state
    )
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(reprs_train, coverages_train, delta=delta, M=M)
    #print(wsc_star, v_star, a_star, b_star)
    # Estimate coverage
    coverage = wsc_vab(reprs_test, coverages_test, v_star, a_star, b_star)
    return coverage


def get_repr_samples_from_dl(model, dl, args):
    reprs_list = []
    # We assume that the dataloader is not shuffled (else `coverages` and `reprs` won't match)
    assert not isinstance(dl.batch_sampler.sampler, torch.utils.data.RandomSampler)
    for batch in dl:
        with torch.no_grad():
            past_events, target_time, target_label = get_history_and_target(batch, args)
            if args.include_poisson:        
                process_keys = list(model.processes.keys())
                # We only need the representations of the base process, not the Poisson. 
                base_model = model.processes[process_keys[0]]
            else:
                base_model = model
            reprs, representations_mask, artifacts = base_model.encode(events=past_events, encoding_type='encoder')
        reprs_list.append(reprs[:, -1, :]) # TODO: check that everything is OK

    reprs = torch.cat(reprs_list, dim=0)
    reprs = reprs.detach().cpu()
    return reprs


def get_logz_samples_from_dl(model, dl, args, nb_samples=500):
    logz_samples_list = []
    for batch in dl:
        past_events, target_time, target_label = get_history_and_target(batch, args)
        logz_samples = get_logz_samples(model, past_events, nb_samples=nb_samples)
        # Due to weird behaviour of the tpps library, we can't use torch.no_grad() and have to detach the tensor instead
        logz_samples = logz_samples.detach().cpu()
        logz_samples_list.append(logz_samples)
    logz_samples = torch.cat(logz_samples_list)
    logz_samples = logz_samples.sort(dim=1).values
    return logz_samples


# def compute_partition_for_repr(model, dl_calib, args, nb_partitions=10):
#     reprs = get_repr_samples_from_dl(model, dl_calib, args)
#     kmeans = KMeans(n_clusters=nb_partitions).fit(reprs)
#     return kmeans


# def compute_partition_for_z(model, dl_calib, args, nb_partitions=10):
#     z_samples = get_z_samples_from_dl(model, dl_calib, args)
#     # The distance between two samples is similar to the profile distance of CD-split+
#     # except that we use \hat{H}^-1 instead of \hat{H} in the definition of the distance.
#     kmeans = KMeans(n_clusters=nb_partitions).fit(z_samples)
#     return kmeans


# def coverage_error_conditional_to_partition(kmeans, coverages, model, dl_test, alpha, args):
#     z_samples = get_z_samples_from_dl(model, dl_test, args)
#     test_partitions = kmeans.predict(z_samples)
#     cond_coverage_list = []
#     for i in range(kmeans.n_clusters):
#         cond_coverages = coverages[test_partitions == i]
#         if len(cond_coverages) == 0: # Special case: no sample in the partition
#             cond_coverage = 0.5
#         else:
#             cond_coverage = cond_coverages.mean()
#         cond_coverage_list.append(cond_coverage)
#     cond_coverages = torch.tensor(cond_coverage_list)
#     error = (cond_coverages - (1 - alpha)).square().mean()
#     return error


class ConditionalCoverageComputer:
    def __init__(self, model, alpha, args):
        self.model = model
        self.alpha = alpha
        self.args = args
    
    def get_partition_features(self, dl_calib):
        pass

    def compute_partition(self, dl_calib, nb_partitions):
        features = self.get_partition_features(dl_calib)
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


class ConditionalCoverageComputerForRepr(ConditionalCoverageComputer):
    def get_partition_features(self, dl_calib):
        return get_repr_samples_from_dl(self.model, dl_calib, self.args)


class ConditionalCoverageComputerForZ(ConditionalCoverageComputer):
    def get_partition_features(self, dl_calib):
        return get_logz_samples_from_dl(self.model, dl_calib, self.args)
