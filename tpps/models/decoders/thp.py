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


class THP(MCDecoder):
    """A mlp decoder based on Monte Carlo estimations. See https://arxiv.org/abs/2002.09291.pdf

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
        super(THP, self).__init__(
            name="thp",
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.w_t = nn.Linear(in_features=1, out_features=marks)
        self.w_h = nn.Linear(in_features=units_mlp[0], out_features=marks)
        self.activation = ParametricSoftplus(units=marks)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            # units_mlp in this class also provides the input dimensionality
            # of the mlp
            input_shape=units_mlp[0],
            activation_final=activation_final_mlp)

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

        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)

        delta_t = (query - prev_times)/prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1).float()

        w_delta_t = self.w_t(delta_t)
        w_history = self.w_h(history_representations)
        outputs = self.activation(w_delta_t + w_history)

        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)

        return th.log(outputs), intensity_mask, artifacts 
