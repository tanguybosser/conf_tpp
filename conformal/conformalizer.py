from abc import abstractmethod
import torch
from torch.distributions import Categorical, Beta
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl

from .dist_util import icdf, get_logz_samples
from .tpp_util import get_history_and_target
from tpps.utils.stability import check_tensor


# ========================= The base class of all predictors =========================

class ConformalizerBase:
    def __init__(self, dl_calib, model, alpha, args):
        self.model = model
        self.alpha = alpha
        self.args = args
        # We can't use torch.no_grad() else there is an error when computing the CDF of the tpps package
        #with torch.no_grad():
        self.scores = [
            self.get_score_on_batch(batch) 
            for batch in dl_calib
        ]
        self.scores = torch.concat(self.scores)
        self.q = self.conformal_quantile(self.scores)
        if self.q == 1:
            print('Warning: q = 1, this will lead to large prediction regions.')

    def get_score_on_batch(self, batch):
        past_events, target_time, target_label = get_history_and_target(batch, self.args)
        return self.get_score(past_events, target_time, target_label)

    @abstractmethod
    def get_score(self, inter_time_dist, inter_times, seq_len):
        pass

    def conformal_quantile(self, scores):
        n = scores.shape[0]
        level = torch.ceil(torch.tensor((1 - self.alpha) * (n + 1))) / (n + 1)
        if level > 1:
            return torch.tensor(self.max_score(), device=scores.device)
        level = level.type(scores.dtype).to(scores.device)
        return torch.quantile(scores, level)
    
    def max_score(self):
        pass

    def get_predictor_type(self):
        return 'joint'

# ========================= Predictors for the time only =========================

class TimeConformalizer(ConformalizerBase):
    def get_time_prediction_region(self, past_events):
        # Must return a batch of list of intervals
        pass
    
    def get_joint_prediction_region(self, past_events):
        nb_labels = past_events.labels.shape[-1]
        time_prediction_regions = self.get_time_prediction_region(past_events)
        
        preds = [
            {label: region for label in range(nb_labels)}
            for region in time_prediction_regions
        ]
        return np.array(preds)

    def get_predictor_type(self):
        return 'time'


class ConformalizerRightBound(TimeConformalizer):
    def get_time_right_bound(self, past_events):
        pass

    def get_time_prediction_region(self, past_events):
        last_observed_times = past_events.times_final
        right_bounds = self.get_time_right_bound(past_events)
        right_bounds = torch.maximum(last_observed_times, right_bounds)
        preds = [
            [[left.item(), right.item()]]
            for left, right in zip(last_observed_times, right_bounds)
        ]
        return np.array(preds)
    
    def max_score(self):
        return torch.inf


class ConformalConstantTime(ConformalizerRightBound):
    def get_score(self, past_events, target_time, target_label):
        last_observed_times = past_events.times_final
        return target_time - last_observed_times
    
    def get_time_right_bound(self, past_events):
        last_observed_times = past_events.times_final
        return last_observed_times + self.q


class CQRL(ConformalizerRightBound):
    def get_unconformalized_right_bound(self, past_events):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]

        coverage_level = torch.full(batch_shape + (1,), 1 - self.alpha, dtype=torch.float32, device=device)
        quantile = icdf(self.model, coverage_level, past_events)[:, 0]
        return quantile

    def get_score(self, past_events, target_time, target_label):
        score = target_time - self.get_unconformalized_right_bound(past_events)
        return score
    
    def get_time_right_bound(self, past_events):
        return self.get_unconformalized_right_bound(past_events) + self.q

class QRL(CQRL):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.zeros(batch_shape, device=self.args.device)

class CQR(TimeConformalizer):
    def get_unconformalized_bounds(self, past_events):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]
        coverage_level_low = torch.full(batch_shape + (1,), self.alpha / 2, dtype=torch.float32, device=device)
        coverage_level_up = torch.full(batch_shape + (1,), 1 - self.alpha / 2, dtype=torch.float32, device=device)
        quantile_low = icdf(self.model, coverage_level_low, past_events)[:, 0]
        quantile_up = icdf(self.model, coverage_level_up, past_events)[:, 0]
        return quantile_low, quantile_up
    
    def get_score(self, past_events, target_time, target_label):
        q_l, q_u = self.get_unconformalized_bounds(past_events)
        score = torch.maximum(q_l - target_time, target_time - q_u)
        return score
    
    def get_bounds(self, past_events):
        q_l, q_u = self.get_unconformalized_bounds(past_events)
        return q_l - self.q, q_u + self.q

    def get_time_prediction_region(self, past_events):
        last_observed_times = past_events.times_final
        left_bounds, right_bounds = self.get_bounds(past_events)
        left_bounds = torch.maximum(last_observed_times, left_bounds)
        right_bounds = torch.maximum(last_observed_times, right_bounds)
        preds = [
            [[left.item(), right.item()]]
            for left, right in zip(left_bounds, right_bounds)
        ]
        return np.array(preds)
    
class QR(CQR):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.zeros(batch_shape, device=self.args.device)


# ========================= Predictors for the mark only =========================


class MarkPredictor(ConformalizerBase):
    def get_mark_prediction_region(self, past_events):
        # Must return a batch of booleans vectors indicating whether the mark is in the prediction region
        pass
    
    def get_joint_prediction_region(self, past_events):
        last_observed_times = past_events.times_final
        mark_preds = self.get_mark_prediction_region(past_events)
        
        preds = []
        for mark_pred, last_observed_time in zip(mark_preds, last_observed_times):
            pred = {
                label: [[last_observed_time.item(), torch.inf]]
                for label, in_region in enumerate(mark_pred)
                if in_region
            }
            if len(pred) == 0:
                print('Warning: Zero-length set encountered')
            preds.append(pred)
        return np.array(preds)
    
    def get_time_sample(self, past_events, nb_samples=100):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]
        epsilon = 1e-4
        alpha = torch.linspace(0 + epsilon, 1 - epsilon, nb_samples, device=device)[None, :].expand(batch_shape + (nb_samples,))
        return icdf(self.model, alpha, past_events)
    
    def get_pmf(self, past_events):
        """Returns a PMF for the next label"""
        time_sample = self.get_time_sample(past_events)
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=time_sample, events=past_events)
        proba = log_mark_density.exp().mean(dim=1) # We average probabilities over samples
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        return proba

    def get_predictor_type(self):
        return 'mark'


class APS(MarkPredictor):
    def get_score(self, past_events, target_time, target_label):
        proba = self.get_pmf(past_events)
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.cumsum(torch.take_along_dim(proba, pi, dim=1), dim=1)

        target_label_index = target_label.argmax(-1)
        score = torch.take_along_dim(srt, pi.argsort(dim=1), dim=1)[range(proba.shape[0]), target_label_index]
        return score

    def get_mark_prediction_region(self, past_events):
        proba = self.get_pmf(past_events)
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.cumsum(torch.take_along_dim(proba, pi, dim=1), dim=1)
        srt_mask = srt <= self.q
        #srt_mask_sum = torch.sum(srt_mask, dim=-1)
        #mask = srt_mask_sum == proba.shape[1]
        #srt_mask_sum[mask] = proba.shape[1]-1
        #srt_mask[torch.arange(srt_mask.shape[0]), srt_mask_sum] = True
        srt_mask[:, 0] = True  #Avoid 0 size sets
        prediction_sets = torch.take_along_dim(srt_mask, pi.argsort(dim=1), dim=1)
        return prediction_sets

    def max_score(self):
        return 1

class NaiveAPS(APS):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.args.device)

class APS2(APS):
    def get_score(self, past_events, target_time, target_label):
        proba = self.get_pmf(past_events)
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        target_label_index = target_label.argmax(-1)
        score_idx = torch.where(pi == target_label_index.unsqueeze(-1))[1]
        rand = torch.rand(n).to(self.args.device) * srt[range(n), score_idx]
        score = torch.cumsum(srt, dim=1)[range(n), score_idx] - rand 
        return score
    
    def get_mark_prediction_region(self, past_events):
        proba = self.get_pmf(past_events)
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        srt_cumsum = torch.cumsum(srt, dim=1)
        rand = torch.rand(n,1).to(self.args.device) * srt
        ind = srt_cumsum - rand <= self.q 
        ind[:,0] = True #Avoids 0 size sets
        prediction_sets = torch.take_along_dim(ind, pi.argsort(dim=1), dim=1)
        return prediction_sets

class NaiveAPS2(APS2):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.args.device)


class RAPS(APS):    
    def get_score(self, past_events, target_time, target_label):
        lambda_reg = self.args.lambda_reg
        k_reg = self.args.k_reg
        proba = self.get_pmf(past_events)
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        reg = torch.tensor(k_reg*[0,] + (proba.shape[1]-k_reg)*[lambda_reg,])[None,:]
        reg = reg.to(self.args.device)
        srt_reg = srt + reg
        target_label_index = target_label.argmax(-1)
        score_idx = torch.where(pi == target_label_index.unsqueeze(-1))[1]
        rand = torch.rand(n).to(self.args.device) * srt_reg[range(n), score_idx]
        score = torch.cumsum(srt_reg, dim=1)[range(n), score_idx] - rand 
        return score
    
    def get_mark_prediction_region(self, past_events):
        lambda_reg = self.args.lambda_reg
        k_reg = self.args.k_reg
        proba = self.get_pmf(past_events)
        n = proba.shape[0]
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        pi = proba.argsort(dim=1, descending=True)
        srt = torch.take_along_dim(proba, pi, dim=1)
        reg = torch.tensor(k_reg*[0,] + (proba.shape[1]-k_reg)*[lambda_reg,])[None,:]
        reg = reg.to(self.args.device)
        srt_reg = srt + reg
        srt_reg_cumsum = torch.cumsum(srt_reg, dim=1)
        rand = torch.rand(n,1).to(self.args.device) * srt_reg
        ind = srt_reg_cumsum - rand <= self.q 
        #srt_mask_sum = torch.sum(ind, dim=-1)
        #mask = srt_mask_sum == proba.shape[1]
        #srt_mask_sum[mask] = proba.shape[1]-1
        #ind[torch.arange(ind.shape[0]), srt_mask_sum] = True 
        ind[:,0] = True #Avoids 0 size sets
        prediction_sets = torch.take_along_dim(ind, pi.argsort(dim=1), dim=1)
        return prediction_sets


class NaiveRAPS(RAPS):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.args.device)


class APSConditionalToSampledTime(APS):
    def get_time_sample(self, past_events):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]
        alpha = torch.rand(batch_shape + (1,), device=device)
        return icdf(self.model, alpha, past_events)


class ConformalizedProb(MarkPredictor):
    def get_score(self, past_events, target_time, target_label):
        proba = self.get_pmf(past_events)
        # Now we have a PMF over labels, thus proba.sum(dim=1) ~= 1
        target_label_index = target_label.argmax(-1)
        score = 1 - proba[range(proba.shape[0]), target_label_index]
        return score

    def get_mark_prediction_region(self, past_events):
        proba = self.get_pmf(past_events)
        prediction_sets = proba >= (1 - self.q) #I think that there is no guarantee against empty prediction sets. 
        return prediction_sets

class ConformalizedProb2(ConformalizedProb):
    def get_mark_prediction_region(self, past_events):
        proba = self.get_pmf(past_events)
        max_prob_idx = torch.argmax(proba, dim=-1)
        prediction_sets = proba >= (1 - self.q) 
        prediction_sets[range(proba.shape[0]), max_prob_idx] = True    
        return prediction_sets


# ========================= Predictors for the time and mark jointly =========================

class JointIndependent(ConformalizerBase):
    def __init__(self, time_predictor, label_predictor):
        self.model = time_predictor.model
        self.alpha = time_predictor.alpha
        self.args = time_predictor.args
        self.time_predictor = time_predictor
        self.label_predictor = label_predictor
        self.q = (self.time_predictor.q, self.label_predictor.q)
        self.scores = (self.time_predictor.scores, self.label_predictor.scores)
    
    def get_joint_prediction_region(self, past_events):
        time_preds = self.time_predictor.get_time_prediction_region(past_events)
        label_preds = self.label_predictor.get_mark_prediction_region(past_events)
        
        preds = []
        for time_pred, label_pred in zip(time_preds, label_preds):
            pred = {
                label: time_pred
                for label, in_region in enumerate(label_pred)
                if in_region
            }
            if len(pred) == 0:
                print('Warning: Zero-length set encountered')
            preds.append(pred)
        return np.array(preds)


class CQRL_RAPS(JointIndependent):
    def __init__(self, dl_calib, model, alpha, args):
        time_predictor = CQRL(dl_calib, model, alpha / 2, args)
        label_predictor = RAPS(dl_calib, model, alpha / 2, args)
        super().__init__(time_predictor, label_predictor)


class Naive_QRL_RAPS(JointIndependent):
    def __init__(self, dl_calib, model, alpha, args):
        time_predictor = QRL(dl_calib, model, alpha / 2, args)
        label_predictor = NaiveRAPS(dl_calib, model, alpha / 2, args)
        super().__init__(time_predictor, label_predictor)


class CHDR(ConformalizerBase):
    def sanity_check_cdf(self, past_events):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]

        time_sample_at_infinity = torch.tensor([[1000.]], device=device).expand(batch_shape + (1,))
        cdf_value = self.model.cdf(query=time_sample_at_infinity, events=past_events)[0]
        valid = (cdf_value >= 0.99).all()
        if not valid:
            print(f'Warning: sanity check failed: cdf(inf) = {cdf_value}')

    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        # Get samples of log(Z) to compute the CDF F_Z
        logz_sample = get_logz_samples(self.model, past_events)
        # Compute density at target_time and target_label
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=target_time.unsqueeze(1), events=past_events)
        log_density = log_density.squeeze(1)
        target_label_index = target_label.argmax(-1).unsqueeze(1)
        true_logz = torch.gather(log_density, 1, target_label_index).squeeze(1)
        # F_Z(f(y))
        cdf_eval = (logz_sample[:, :] <= true_logz[:, None]).float().mean(-1)
        # Final score
        score = 1 - cdf_eval
        return score
    
    def create_interval(self, seq):
        # Return the intervals of True values in the sequence
        seq = torch.cat([torch.tensor([False], device=seq.device), seq, torch.tensor([False], device=seq.device)]).int()
        diffs = torch.diff(seq)
        starts = (diffs == 1).nonzero(as_tuple=False).squeeze(-1)
        ends = (diffs == -1).nonzero(as_tuple=False).squeeze(-1) - 1
        return torch.stack((starts, ends + 1), dim=1)

    def get_time_range(self, method, past_events, nb_steps):
        batch_shape = past_events.times.shape[:1]
        device = self.args.device

        epsilon = 1e-4 # TODO: different datasets may require different epsilon
        # epsilon must be >= 1e-4 on reddit
        # epsilon should probably be chosen to be closer to 1e-7 on mooc
        if method == 'linear':
            last_time = past_events.times_final
            delta_range = torch.linspace(0 + epsilon, 0.0004, nb_steps, device=device)[None, :].expand(batch_shape + (nb_steps,))
            time_range = last_time[:, None] + delta_range
        elif method in ['sampling', 'beta_sampling']:
            if method == 'sampling':
                alpha = torch.linspace(0 + epsilon, 1 - epsilon, nb_steps, device=device)
            elif method == 'beta_sampling':
                alpha = Beta(0.8, 0.8).sample((nb_steps,)).to(device).sort().values
                alpha += epsilon
            alpha = alpha[None, :].expand(batch_shape + (nb_steps,))
            time_range = icdf(self.model, alpha, past_events)
            # Note that alpha ~= self.model.cdf(query=time_range, events=past_events)

        # On mooc, there is not enough precision to distinguish successive time values
        # Look at the values of time_range[0].detach().cpu().numpy()
        # For example, np.array([3.6464164, 3.6464165, 3.6464166, 3.6464167], dtype=np.float32) contains duplicates

        return time_range

    def sanity_check_density(self, time_range, log_density):
        delta = time_range[:,1:] - time_range[:,:-1]
        check_tensor(delta, positive=True)
        density = torch.exp(log_density)
        integrals = torch.trapz(density, time_range[:, :, None], dim=1).sum(dim=1)
        # The integrals should be close to 1
        print('Integrals', integrals)

    def get_joint_prediction_region(self, past_events):
        batch_shape = past_events.times.shape[:1]
        device = self.args.device
        nb_labels = past_events.labels.shape[-1]
        # Get samples of log(Z) to compute the CDF F_Z
        logz_sample = get_logz_samples(self.model, past_events)
        # Quantile alpha of F_Z
        self.q = self.q.type(logz_sample.dtype)
        logz_quantile = torch.quantile(logz_sample, 1 - self.q, dim=1)
        # We determine the best times to evaluate the density from which the intervals will be created.
        # This is important since the density is often restricted to a very small region.
        nb_steps = 300
        time_range = self.get_time_range('beta_sampling', past_events, nb_steps=nb_steps)
        time_range, _ = torch.sort(time_range, dim=-1) 
        # Density evaluation
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=time_range, events=past_events)
        assert log_density.shape == (batch_shape[0], nb_steps, nb_labels)
        #self.sanity_check_density(time_range, log_density)
        # Thresholding
        valid = log_density >= logz_quantile[:, None, None]
        # Intervals are created from regions that are marked as valid
        preds = []
        for b in range(valid.shape[0]):
            pred = {}
            for mark in range(valid.shape[2]):
                seq = valid[b, :, mark]
                index_intervals = self.create_interval(seq)
                index_intervals = torch.minimum(index_intervals, torch.tensor(nb_steps - 1, device=device))
                intervals = time_range[b][index_intervals]
                pred[mark] = intervals.detach().cpu().tolist()
            preds.append(pred)
        return preds

    def max_score(self):
        return 1


class HDR(CHDR):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.args.device)


class CHDRTime(TimeConformalizer, CHDR):
    def get_logz_samples(self, past_events, nb_samples=500):
        device = self.args.device
        batch_shape = past_events.times.shape[:1]

        # Sample quantile level
        alpha = torch.rand(batch_shape + (nb_samples,), device=device, dtype=torch.float64)
        alpha = alpha.sort(dim=1).values
        #self.sanity_check_cdf(past_events)
        # Sample time
        time_sample = icdf(self.model, alpha, past_events)
        # Sample label
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=time_sample, events=past_events)
        # Sample Z
        logz_sample = torch.logsumexp(log_density, dim=-1)
        assert logz_sample.shape == batch_shape + (nb_samples,)
        return logz_sample
    
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        # Get samples of log(Z) to compute the CDF F_Z
        logz_sample = self.get_logz_samples(past_events)
        # Compute density at target_time and target_label
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=target_time.unsqueeze(1), events=past_events)
        log_ground_density = torch.logsumexp(log_density, dim=-1)
        log_ground_density = log_ground_density.squeeze(1)
        # F_Z(f(y))
        cdf_eval = (logz_sample[:, :] <= log_ground_density[:, None]).float().mean(-1)
        # Final score
        score = 1 - cdf_eval
        return score

    def get_time_prediction_region(self, past_events):
        batch_shape = past_events.times.shape[:1]
        device = self.args.device
        # Get samples of log(Z) to compute the CDF F_Z
        logz_sample = self.get_logz_samples(past_events)
        # Quantile alpha of F_Z
        self.q = self.q.type(logz_sample.dtype)
        logz_quantile = torch.quantile(logz_sample, 1 - self.q, dim=1)
        # We determine the best times to evaluate the density from which the intervals will be created.
        # This is important since the density is often restricted to a very small region.
        nb_steps = 300
        time_range = self.get_time_range('beta_sampling', past_events, nb_steps=nb_steps)
        time_range, _ = torch.sort(time_range, dim=-1) 
        # Density evaluation
        log_density, log_mark_density, y_pred_mask = self.model.log_density(query=time_range, events=past_events)
        log_ground_density = torch.logsumexp(log_density, dim=-1)
        assert log_ground_density.shape == (batch_shape[0], nb_steps)
        # Thresholding
        valid = log_ground_density >= logz_quantile[:, None]
        # Intervals are created from regions that are marked as valid
        preds = []
        for b in range(valid.shape[0]):
            seq = valid[b, :]
            index_intervals = self.create_interval(seq)
            index_intervals = torch.minimum(index_intervals, torch.tensor(nb_steps - 1, device=device))
            intervals = time_range[b][index_intervals]
            pred = intervals.detach().cpu().tolist()
            preds.append(pred)
        return preds

    def max_score(self):
        return 1


class HDRTime(CHDRTime):
    def get_score(self, past_events, target_time, target_label):
        batch_shape = past_events.times.shape[:1]
        return torch.full(batch_shape, 1 - self.alpha, device=self.args.device)


class CHDR_RAPS(JointIndependent):
    def __init__(self, dl_calib, model, alpha, args):
        time_predictor = CHDRTime(dl_calib, model, alpha / 2, args)
        label_predictor = RAPS(dl_calib, model, alpha / 2, args)
        super().__init__(time_predictor, label_predictor)


class Naive_HDR_RAPS(JointIndependent):
    def __init__(self, dl_calib, model, alpha, args):
        time_predictor = HDRTime(dl_calib, model, alpha / 2, args)
        label_predictor = NaiveRAPS(dl_calib, model, alpha / 2, args)
        super().__init__(time_predictor, label_predictor)
