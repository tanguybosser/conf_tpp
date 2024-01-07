import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, subtract_exp, check_tensor

class SepRMTPPIndDecoder(VariableHistoryDecoder):
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
            **kwargs):
        if encoding not in ["times_only", "log_times_only"]:
            raise ValueError("Wrong encoding for RMTPP decoder")
        super(SepRMTPPIndDecoder, self).__init__(
            name="rmtpp",
            input_size=units_mlp[0],
            encoding=encoding,
            marks=marks)
        self.w = nn.Parameter(th.Tensor(1))
        self.w_h = nn.Linear(self.input_size, marks)
        self.w_t  = nn.Linear(self.input_size, 1)
        self.multi_labels = multi_labels
        self.reset_parameters()

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

        #v_h_t = query_representations[:, :, 0]                        # [B,T] v_t * h_j #query representations are in fact history     embeddings. 
        v_h_t = self.w_t(history_representations_time)                         #[B,T,1]
        v_h_t = v_h_t.squeeze()                                         #[B,T]

        #v_h_m = query_representations[:, :, 1:]                       # [B,T,M] V^k * h_j
        v_h_m = self.w_h(history_representations_mark)                         #[B,T,M]

        w_delta_t = self.w * (query - prev_times)                     # [B,T]
        
        base_log_intensity = v_h_t + w_delta_t                        # [B,T]


        if self.multi_labels:
            p_m = th.sigmoid(v_h_m)                                   # [B,T,M]
        else:
            p_m = th.softmax(v_h_m, dim=-1)                           # [B,T,M]
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
        marked_intensity_itg = marked_intensity_itg * p_m             # [B,T,M]

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
