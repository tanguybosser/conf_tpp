#event_times [B,L]
import pdb
from pandas import array
import torch as th
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tpps.utils.events import Events
from tpps.models.base.process import Process

def monte_carlo_integration(u:th.Tensor, tn:th.Tensor ,density:th.Tensor, n:int):
    power = float(3)
    MC = ((1 + (tn-1)*u)/th.pow(input=u , exponent=power)) * density #[N*B, L]
    b, l = int(MC.shape[0]/n), MC.shape[1]
    MC = th.reshape(MC, (b, -1 , l))
    MC = th.sum(MC, dim=1)
    MC = MC/n
    return MC

