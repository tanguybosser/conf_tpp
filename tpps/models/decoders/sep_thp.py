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


class SepTHP(MCDecoder):
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

            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 2,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            **kwargs):
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(SepTHP, self).__init__(
            name="sep-thp",
            input_size=units_mlp[0],
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.w_t = nn.Linear(in_features=1, out_features=1)
        self.w_h = nn.Linear(in_features=units_mlp[0], out_features=1)
        self.activation = ParametricSoftplus(units=1)
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.hist_time_grouping = hist_time_grouping
        if self.hist_time_grouping == 'summation':
            self.marks1 = nn.Linear(
            in_features=units_mlp[0], out_features=units_mlp[1])
            self.mark_time = nn.Linear(
                in_features=self.encoding_size, out_features=units_mlp[1]
            )
        elif self.hist_time_grouping == 'concatenation':
            self.mark_time = nn.Linear(
                in_features=self.encoding_size + self.input_size, out_features=units_mlp[1]
            )
        self.mark_activation = self.get_mark_activation(mark_activation)

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
        
        (query_representations,
         intensity_mask) = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations, 
            representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        b,l = query.shape
        representations_time = representations[0:b,:,:]
        representations_mark = representations[b:,:,:]

        history_representations_time = take_3_by_2(                            
            representations_time, index=prev_times_idxs)                   # [B,T,D]
        history_representations_mark = take_3_by_2(                          
            representations_mark, index=prev_times_idxs)


        if self.hist_time_grouping == 'summation':
            p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.marks1(history_representations_mark) + self.mark_time(query_representations))), dim=-1) 
        elif self.hist_time_grouping == 'concatenation':
            history_times = th.cat((history_representations_mark, query_representations), dim=-1)
            p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_times))), dim=-1)
        
        p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)

        prev_times = prev_times + epsilon(dtype=prev_times.dtype, device=prev_times.device)

        delta_t = (query - prev_times)/prev_times
        check_tensor(delta_t)
        delta_t = delta_t.unsqueeze(-1)

        w_delta_t = self.w_t(delta_t) #[B,T,1]
        w_history = self.w_h(history_representations_time) #[B,T,1]
        outputs = self.activation(w_delta_t + w_history) #[B,T,1]

        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)

        outputs = outputs * p_m

        return th.log(outputs), intensity_mask, artifacts 
    


    
    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation
