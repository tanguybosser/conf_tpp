import pdb

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpps.models.decoders.hawkes import HawkesDecoder
from tpps.utils.events import Events
from tpps.utils.nnplus import non_neg_param
from tpps.processes.hawkes_fast import decoder_fast as hawkes_decoder
# from tpp.processes.hawkes_slow import decoder_slow as hawkes_decoder


class HawkesFixedDecoder(HawkesDecoder):
    """A parametric Hawkes Process decoder with pre-fixed parameters.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self, 
                 baselines,
                 decays,
                 adjacency,
                 marks: Optional[int] = 1, 
                 **kwargs):
        super(HawkesFixedDecoder, self).__init__(name="hawkes_fixed", marks=marks)
        self.alpha = nn.Parameter(th.Tensor(adjacency), requires_grad=False)
        self.beta = nn.Parameter(th.Tensor(decays), requires_grad=False)
        self.mu = nn.Parameter(th.Tensor(baselines), requires_grad=False)