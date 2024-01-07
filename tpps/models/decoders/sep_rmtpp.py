import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, subtract_exp, check_tensor

class SepRMTPPDecoder(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            **kwargs):
        super(SepRMTPPDecoder, self).__init__(
            name="sep-rmtpp",
            input_size=units_mlp[0],
            encoding=encoding,
            marks=marks)
        self.w = nn.Parameter(th.Tensor(1))
        self.w_h = nn.Linear(self.input_size, marks)
        self.w_t  = nn.Linear(self.input_size, 1)
        self.marks2 = nn.Linear(
            in_features=units_mlp[1], out_features=marks)
        self.multi_labels = multi_labels
        self.reset_parameters()
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

        
    def reset_parameters(self):
        nn.init.uniform_(self.w, b=0.001)

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            sampling: Optional[bool] = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the intensities for each query time given event
        representations.

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
            representations: [B,L+1,D] Representations of window start and
                each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

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

        b,l = query.shape
        representations_time = representations[0:b,:,:]
        representations_mark = representations[b:,:,:]
        history_representations_time = take_3_by_2(                            
            representations_time, index=prev_times_idxs)                   # [B,T,D]
        history_representations_mark = take_3_by_2(                          
            representations_mark, index=prev_times_idxs)
        v_h_t = self.w_t(history_representations_time)                         #[B,T,1]
        v_h_t = v_h_t.squeeze()                                         #[B,T]
        
        w_delta_t = self.w * (query - prev_times)                     # [B,T]                    #[B,T]
        
        base_log_intensity = v_h_t + w_delta_t                        # [B,T]


        if self.hist_time_grouping == 'summation':
            p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.marks1(history_representations_mark) + self.mark_time(query_representations))), dim=-1) 
        elif self.hist_time_grouping == 'concatenation':
            history_times = th.cat((history_representations_mark, query_representations), dim=-1)
            p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_times))), dim=-1)
        
        regulariser = epsilon(dtype=p_m.dtype, device=p_m.device)
        p_m = p_m + regulariser

        marked_log_intensity = base_log_intensity.unsqueeze(
            dim=-1)  # [B,T,1]
        marked_log_intensity = marked_log_intensity + th.log(p_m)     # [B,T,M]

        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        exp_1, exp_2 = v_h_t + w_delta_t, v_h_t                         # [B,T]
        # Avoid exponentiating to get masked infinity
        exp_1, exp_2 = exp_1 * intensity_mask, exp_2 * intensity_mask   # [B,T]
        base_intensity_itg = subtract_exp(exp_1, exp_2)
        base_intensity_itg = base_intensity_itg / self.w                # [B,T]
        base_intensity_itg = th.relu(base_intensity_itg)
    
        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)   # [B,T,1]
        
        ones = th.ones_like(p_m)
        marked_intensity_itg = (marked_intensity_itg / self.marks) * ones #[B,T,M]

        artifacts_decoder = {
            "base_log_intensity": base_log_intensity,
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        check_tensor(marked_log_intensity)
        check_tensor(marked_intensity_itg * intensity_mask.unsqueeze(-1),
                     positive=True)
        return (marked_log_intensity,
                marked_intensity_itg,
                intensity_mask,
                artifacts)                      # [B,T,M], [B,T,M], [B,T], Dict


    def get_mark_activation(self, mark_activation):
        if mark_activation == 'relu':
            mark_activation = th.relu
        elif mark_activation == 'tanh':
            mark_activation = th.tanh
        elif mark_activation == 'sigmoid':
            mark_activation = th.sigmoid
        return mark_activation