import torch as th
import torch.nn.functional as F
import torch.nn as nn

from tpps.pytorch.activations import ParametricSoftplus

from typing import List, Optional, Tuple, Dict

from tpps.models.decoders.base.monte_carlo import MCDecoder
from tpps.models.base.process import Events

from tpps.pytorch.models import MLP

from tpps.utils.encoding import encoding_size
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.pytorch.layers.dense import NonNegLinear

class SAHP(MCDecoder):
    """A mlp decoder based on Monte Carlo estimations. See https://arxiv.org/pdf/1907.07561.pdf

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either `None`, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        emb_dim: Size of the embeddings (default=2).
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(SAHP, self).__init__(
            name="sahp",
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=marks, bias=False)
        self.eta = nn.Linear(in_features=units_mlp[0], out_features=marks, bias=False)
        self.gamma = nn.Linear(in_features=units_mlp[0], out_features=marks, bias=False)
        self.activation = nn.GELU()
        self.activation_gamma = ParametricSoftplus(units=marks)
        self.final_activation = ParametricSoftplus(units=marks)

    def log_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the log_intensity and a mask

        Args:
            events: [B,L] Times and labels of events.
            query: [B,T] Times to evaluate the intensity function.
            prev_times: [B,T] Times of events directly preceding queries.
            prev_times_idxs: [B,T] Indexes of times of events directly
                preceding queries. These indexes are of window-prepended
                events.
            pos_delta_mask: [B,T] A mask indicating if the time difference
                `query - prev_times` is strictly positive.
            is_event: [B,T] A mask indicating whether the time given by
                `prev_times_idxs` corresponds to an event or not (a 1 indicates
                an event and a 0 indicates a window boundary).
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures.

        """
        
        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        mu = self.activation(self.mu(history_representations)) #[B,T,K]
        eta = self.activation(self.eta(history_representations))
        gamma = self.activation_gamma(self.gamma(history_representations))
        check_tensor(gamma, positive=True)
        delta_t = (query - prev_times) * pos_delta_mask
        #delta_t = (query - prev_times)

        delta_t = delta_t + epsilon(dtype=delta_t.dtype, device=delta_t.device)
        delta_t = delta_t.unsqueeze(-1)

        outputs = mu + (eta -mu)*th.exp(-gamma*(delta_t)) #[B,T,K]
        #if th.isinf(th.exp(-gamma*(delta_t))).sum() != 0:
        #    print(delta_t)
        #check_tensor(th.exp(-gamma*(delta_t)))
        #check_tensor(mu)
        #check_tensor(eta)
        outputs = self.final_activation(outputs)

        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)
        check_tensor(outputs, positive=True)
        
        #print(outputs.sum(-1))
        #delta = outputs.sum(-1)[:,:-1:]-outputs[:,1:,:].sum(-1) 
        #print(delta)
        #check_tensor(delta, positive=True)
        return th.log(outputs), intensity_mask, artifacts 
