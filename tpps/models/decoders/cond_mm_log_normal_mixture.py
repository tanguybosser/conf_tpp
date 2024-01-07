import torch as th
import math
import torch.nn as nn

from typing import Dict, Optional, Tuple, List

from tpps.models.decoders.base.variable_history import VariableHistoryDecoder
from tpps.pytorch.models import LAYER_CLASSES

from tpps.utils.events import Events
from tpps.utils.index import take_3_by_2, take_2_by_2
from tpps.utils.stability import epsilon, check_tensor
from tpps.utils.encoding import encoding_size

class CondMarkMixtureLogNormalMixtureDecoder(VariableHistoryDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    to train the model.
    See https://arxiv.org/pdf/1909.12127.pdf.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            n_mixture: int,
            n_mark_mixture: int, 
            units_mlp: List[int],
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            encoding: Optional[str] = "times_only",
            embedding_constraint: Optional[str] = None,
            emb_dim: Optional[int] = 2,
            mark_activation: Optional[str] = 'relu',
            hist_time_grouping: Optional[str] = 'summation',
            **kwargs):
        super(CondMarkMixtureLogNormalMixtureDecoder, self).__init__(
            name="cond-mm-log-normal-mixture",
            input_size=units_mlp[0],
            marks=marks,
            encoding=encoding, 
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,  
            **kwargs)
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        self.mu = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        self.s = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        self.w = nn.Linear(in_features=units_mlp[0], out_features=n_mixture)
        self.hist_time_grouping = hist_time_grouping
        if self.hist_time_grouping == 'summation':
            self.pmc_hist_in = nn.Linear(
            in_features=units_mlp[0], out_features=n_mixture*units_mlp[1])
            self.pmc_time_in = nn.Linear(
            in_features=self.encoding_size, out_features=n_mixture*units_mlp[1])
        elif self.hist_time_grouping == 'concatenation':
            self.pmc_hist_time_in = nn.Linear(
            in_features=units_mlp[0]+self.encoding_size, out_features=n_mixture*units_mlp[1])
        #self.pm_hist_in = nn.Linear(
        #    in_features=units_mlp[0], out_features=self.marks)
        #self.pm_time_in = nn.Linear(
        #    in_features=self.encoding_size, out_features=self.marks
        #    )
        
        
        
        self.pmc_out = nn.Linear(
            in_features=units_mlp[1], out_features=self.marks)
        
        ######### TO ENHANCE FLEXIBILITY?########
        #self.pmc_out = nn.Linear(
        #    in_features=n_mixture*units_mlp[1], out_features=self.marks)
        ###########

        #self.pm_out = nn.Linear(
        #    in_features=1, out_features=n_mark_mixture)
        
        #self.pc_hist_in = nn.Linear(
        #    in_features=units_mlp[0], out_features=units_mlp[1])
        #self.pc_time_in = nn.Linear(
        #    in_features=self.encoding_size, out_features=units_mlp[1]
        #    )
        #self.pc_out = nn.Linear(
        #    in_features=units_mlp[1], out_features=n_mixture)
        '''
        self.hist_time_grouping = hist_time_grouping
        
        print(self.hist_time_grouping)
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
        self.multi_labels = multi_labels
        '''
        self.mark_activation = self.get_mark_activation(mark_activation)
        #self.n_mark_mixture = n_mark_mixture
        self.n_mixture = n_mixture
        self.hidden = units_mlp[1]

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
        
        query.requires_grad = True

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

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)  # [B,T,D] actually history 
        b,l = history_representations.shape[0], history_representations.shape[1]
        delta_t = query - prev_times  # [B,T]
        delta_t = delta_t.unsqueeze(-1)  # [B,T,1]
        delta_t = th.relu(delta_t) #Just to ensure that the deltas are positive ? 
        delta_t = delta_t + (delta_t == 0).float() * epsilon(
        dtype=delta_t.dtype, device=delta_t.device)
        delta_t = th.log(delta_t)

        mu = self.mu(history_representations)  # [B,T,C]
        std = th.exp(self.s(history_representations))
        w = th.softmax(self.w(history_representations), dim=-1) #[B,T,C]

        check_tensor(history_representations)
        check_tensor(query_representations)

        if self.hist_time_grouping == 'summation':
            p_m_c = self.mark_activation(self.pmc_hist_in(history_representations) + self.pmc_time_in(query_representations)) #[B, T, C*HIDDEN]
        elif self.hist_time_grouping == 'concatenation':
            history_times = th.cat((history_representations, query_representations), dim=-1)
            p_m_c = self.mark_activation(self.pmc_hist_time_in(history_times)) #[B,T,C*HIDDEN]



        #p_m_c = self.mark_activation(self.pmc_hist_in(history_representations) + self.pmc_time_in(query_representations)) #[B, T, HIDDEN]
        #p_m_c = p_m_c.unsqueeze(-1) #[B, T, C, 1]
        p_m_c = p_m_c.view(b,l,self.n_mixture, self.hidden) #[B,T,C,HIDDEN]
        p_m_c = self.pmc_out(p_m_c) #[B,T,C, K]
        p_m_c = th.softmax(p_m_c, dim=-1) #[B,T,C,K] 

        #p_m_c = self.mark_activation(self.pm_hist_in(history_representations) + self.pm_time_in(query_representations)) #[B, T, K]
        #p_m_c = p_m_c.unsqueeze(-1) #[B, T, K, 1]

        #p_m_c = th.softmax(
        #    self.pm_out(p_m_c), dim=-1) #[B,T,K,C] 
        #print(th.sum(p_m_c, dim=-1).squeeze(-1), 'pmc')
        '''
        elif self.hist_time_grouping == 'concatenation':
            history_times = th.cat((history_representations, query_representations), dim=-1)
            p_m = th.softmax(
                self.marks2(
                    self.mark_activation(self.mark_time(history_times))), dim=-1)
        '''
        check_tensor(p_m_c, positive=True)
        #p_c = th.softmax(
        #    self.pc_out(
        #    self.mark_activation(self.pc_hist_in(history_representations) + self.pc_time_in(query_representations))
        #    ), dim=-1
        #) #[B,T,C]
        #p_c = p_c.unsqueeze(-1).repeat(1, 1, 1, self.marks) #[B,T,C,K]
        #p_c = w.unsqueeze(-1) #[B,T,C,1]
        #p_m = p_m_c  #[B,T,C,K]
        #p_m = th.sum(p_m, dim=-2).squeeze(-2) #[B,T,K]
        #p_m = p_m + epsilon(dtype=p_m.dtype, device=p_m.device)

        cum_f_c = w * 0.5 * (
                1 + th.erf((delta_t - mu) / (std * math.sqrt(2)))) #[B,T,C]
        cum_f_c = th.clamp(cum_f_c, max=1-1e-6)
        cum_f = th.clamp(th.sum(cum_f_c, dim=-1), max=1-1e-6) #[B,T]
        one_min_cum_f = 1. - cum_f 
        one_min_cum_f = th.relu(one_min_cum_f) + epsilon(
            dtype=cum_f_c.dtype, device=cum_f_c.device)
        
        grad_outputs = th.zeros_like(cum_f_c, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=cum_f_c,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        f_c = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs),
            retain_graph=True,
            create_graph=True)[0]
        '''
        f_c = th.autograd.grad(
            outputs=cum_f_c,
            inputs=query,
            grad_outputs=th.ones_like(cum_f_c),
            retain_graph=True,
            create_graph=True)[0] #[B,T,C]
        '''
        query.requires_grad = False 
        f_c = f_c + epsilon(dtype=f_c.dtype, device=f_c.device)
        assert(th.sum(th.isinf(f_c)) == 0)
        
        f_c = f_c.unsqueeze(-1) #[B,T,C,1]
        
        #print(f_c.shape)
        #print(p_m_c.shape)
        
        joint_density = f_c * p_m_c #[B,T,C,K]
        joint_density = th.sum(joint_density, dim=-2) #[B,T,K]
        f = th.clamp(th.sum(joint_density, dim=-1), 1e-6).unsqueeze(-1) #[B,T,1]
        p_m = joint_density/f #[B,T,K]
        p_m = p_m + epsilon(eps=1e-8, dtype=p_m.dtype, device=p_m.device)
        #check_tensor(p_m * intensity_mask.unsqueeze(-1), positive=True, strict=True)

        base_log_intensity = th.log(f.squeeze(-1) / one_min_cum_f)
        
        assert(th.sum(th.isinf(base_log_intensity)) == 0)
        marked_log_intensity = base_log_intensity.unsqueeze(
            dim=-1)  # [B,T,1]
        check_tensor(th.log(p_m) * intensity_mask.unsqueeze(-1))
        marked_log_intensity = marked_log_intensity + th.log(p_m)  # [B,T,M]

        base_intensity_itg = - th.log(one_min_cum_f)
        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)  # [B,T,1]
        
        #Trick to have Lambda(t) = \sum_k(Lambda_k(t))
        ones = th.ones_like(p_m)
        marked_intensity_itg = (marked_intensity_itg / self.marks) * ones  
        
        
        #marked_intensity_itg = marked_intensity_itg * p_m  # [B,T,M] #IF AND ONLY IF MARKS ARE CONDITIONALLY INDEPENDENT OF TIME !
        

        #intensity_mask = pos_delta_mask  # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        artifacts_decoder = {
            "base_log_intensity": base_log_intensity,
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        check_tensor(marked_log_intensity * intensity_mask.unsqueeze(-1))
        check_tensor(marked_intensity_itg * intensity_mask.unsqueeze(-1),
                     positive=True)
        artifacts["mu"] = mu.detach()[:,-1,:].squeeze().cpu().numpy() #Take history representation of window, i.e. up to last observed event. 
        artifacts["sigma"] = std.detach()[:,-1,:].squeeze().cpu().numpy()
        artifacts["w"] = w.detach()[:,-1,:].squeeze().cpu().numpy()
        artifacts['pmc'] = p_m_c.detach()[0,:5,:,:].squeeze(0).cpu().numpy()
        artifacts['pm'] = p_m.detach()[0,:5,:].squeeze(0).cpu().numpy()

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