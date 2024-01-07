from argparse import Namespace
from torch import nn
from pprint import pprint

from tpps.models.base.enc_dec import EncDecProcess
from tpps.models.base.modular import ModularProcess
from tpps.models.poisson import PoissonProcess

from tpps.models.encoders.base.encoder import Encoder
from tpps.models.encoders.gru import GRUEncoder
from tpps.models.encoders.gru_fixed import GRUFixedEncoder 
from tpps.models.encoders.identity import IdentityEncoder
from tpps.models.encoders.constant import ConstantEncoder
from tpps.models.encoders.mlp_fixed import MLPFixedEncoder
from tpps.models.encoders.mlp_variable import MLPVariableEncoder
from tpps.models.encoders.stub import StubEncoder
from tpps.models.encoders.self_attention import SelfAttentionEncoder
from tpps.models.encoders.self_attention_fixed import SelfAttentionFixedEncoder

from tpps.models.decoders.base.decoder import Decoder
from tpps.models.decoders.conditional_poisson import ConditionalPoissonDecoder
from tpps.models.decoders.conditional_poisson_cm import ConditionalPoissonCMDecoder
from tpps.models.decoders.hawkes import HawkesDecoder
from tpps.models.decoders.hawkes_fixed import HawkesFixedDecoder
from tpps.models.decoders.log_normal_mixture import LogNormalMixtureDecoder
from tpps.models.decoders.mlp_cm import MLPCmDecoder
from tpps.models.decoders.mlp_mc import MLPMCDecoder
from tpps.models.decoders.neural_hawkes import NeuralHawkesDecoder
from tpps.models.decoders.poisson import PoissonDecoder
from tpps.models.decoders.rmtpp import RMTPPDecoder
from tpps.models.decoders.sep_rmtpp import SepRMTPPDecoder
from tpps.models.decoders.sep_rmtpp_ch import SepRMTPPCommonHistDecoder
from tpps.models.decoders.sep_rmtpp_ind import SepRMTPPIndDecoder
from tpps.models.decoders.rmtpp_cm import RMTPPCmDecoder
from tpps.models.decoders.self_attention_cm import SelfAttentionCmDecoder
from tpps.models.decoders.self_attention_simple_cm import SelfAttentionCmDecoder as SelfAttentionSimpleCmDecoder
from tpps.models.decoders.self_attention_mc import SelfAttentionMCDecoder
from tpps.models.decoders.cond_log_normal_mixture import CondLogNormalMixtureDecoder
from tpps.models.decoders.sep_log_normal_mixture import SepLogNormalMixtureDecoder
from tpps.models.decoders.cond_mm_log_normal_mixture import CondMarkMixtureLogNormalMixtureDecoder
from tpps.models.decoders.sep_cond_log_normal_mixture import SepCondLogNormalMixtureDecoder
from tpps.models.decoders.thp import THP
from tpps.models.decoders.sep_thp import SepTHP
from tpps.models.decoders.sep_thp_ch import SepTHPCommonHist
from tpps.models.decoders.sep_thp_mix import SepTHPMix
from tpps.models.decoders.thp_mix import THPMix
from tpps.models.decoders.sahp import SAHP
from tpps.models.decoders.sep_sahp import SepSAHP
from tpps.models.decoders.sahp_mix import SAHPMix
from tpps.models.decoders.sep_sahp_ch import SepSAHPCommonHist
from tpps.models.decoders.sep_sahp_mix import SepSAHPMix
from tpps.models.decoders.sep_mlp_cm import SepMLPCmDecoder
from tpps.models.decoders.sep_mlp_cm_mix import SepMLPCmMixDecoder
from tpps.models.decoders.mlp_cm_mix import MLPCmMixDecoder
from tpps.models.decoders.lnm_joint import LogNormalMixtureDecoderJoint
from tpps.models.decoders.sep_lnm_joint import SepLogNormalMixtureDecoderJoint


ENCODER_CLASSES = {
    "gru": GRUEncoder,
    "gru-fixed": GRUFixedEncoder,
    "identity": IdentityEncoder,
    "constant": ConstantEncoder,
    "mlp-fixed": MLPFixedEncoder,
    "mlp-variable": MLPVariableEncoder,
    "stub": StubEncoder,
    "selfattention": SelfAttentionEncoder,
    "selfattention-fixed": SelfAttentionFixedEncoder}
DECODER_CLASSES = {
    "conditional-poisson": ConditionalPoissonDecoder,
    "conditional-poisson-cm": ConditionalPoissonCMDecoder,
    "hawkes": HawkesDecoder,
    "hawkes_fixed": HawkesFixedDecoder,
    "log-normal-mixture": LogNormalMixtureDecoder,
    "cond-log-normal-mixture": CondLogNormalMixtureDecoder,
    "cond-mm-log-normal-mixture": CondMarkMixtureLogNormalMixtureDecoder, 
    "sep-log-normal-mixture": SepLogNormalMixtureDecoder, 
    #"cond-sep-log-normal-mixture": CondSepLogNormalMixtureDecoder,
    "sep-cond-log-normal-mixture": SepCondLogNormalMixtureDecoder, 
    "mlp-cm": MLPCmDecoder,
    "mlp-mc": MLPMCDecoder,
    "neural-hawkes": NeuralHawkesDecoder,
    "poisson": PoissonDecoder,
    "rmtpp": RMTPPDecoder,
    "sep-rmtpp":SepRMTPPDecoder,
    "sep-rmtpp-ch":SepRMTPPCommonHistDecoder,
    "sep-rmtpp-ind":SepRMTPPIndDecoder,
    "rmtpp-cm": RMTPPCmDecoder,
    "selfattention-cm": SelfAttentionCmDecoder,
    "selfattention-simple-cm": SelfAttentionSimpleCmDecoder,
    "selfattention-mc": SelfAttentionMCDecoder,
    "thp": THP,
    "sep-thp": SepTHP,
    "sep-thp-ch": SepTHPCommonHist,
    "sep-thp-mix": SepTHPMix,
    "thp-mix": THPMix,
    "sahp":SAHP,
    "sep-sahp":SepSAHP,
    "sep-sahp-ch":SepSAHPCommonHist,
    "sep-sahp-mix":SepSAHPMix,
    "sahp-mix":SAHPMix,
    "sep-mlp-cm":SepMLPCmDecoder,
    "sep-mlp-cm-mix":SepMLPCmMixDecoder,
    "mlp-cm-mix":MLPCmMixDecoder,
    "lnm-joint": LogNormalMixtureDecoderJoint,
    "sep-lnm-joint": SepLogNormalMixtureDecoderJoint}


ENCODER_NAMES = sorted(list(ENCODER_CLASSES.keys()))
DECODER_NAMES = sorted(list(DECODER_CLASSES.keys()))

CLASSES = {"encoder": ENCODER_CLASSES, "encoder_histtime": ENCODER_CLASSES, "encoder_histmark": ENCODER_CLASSES, "decoder": DECODER_CLASSES}
NAMES = {"encoder": ENCODER_NAMES, "encoder_time": ENCODER_NAMES, "encoder_mark": ENCODER_NAMES, "decoder": DECODER_NAMES}


def instantiate_encoder_or_decoder(
        args: Namespace, component="encoder") -> nn.Module:
    assert component in {"encoder", "encoder_histtime", "encoder_histmark", "decoder"}
    prefix, classes = component + '_', CLASSES[component]
    if component in ["encoder_histtime", "encoder_histmark"]:
        prefix = 'encoder_'
        kwargs = {
            k[len(prefix):]: v for
            k, v in args.__dict__.items() if k.startswith(prefix)}
        if component == 'encoder_histtime':
            kwargs.update({"encoding":args.encoder_histtime_encoding})
        else:
            print(args.encoder_histmark_encoding)
            kwargs.update({"encoding":args.encoder_histmark_encoding})
    else:
        kwargs = {
            k[len(prefix):]: v for
            k, v in args.__dict__.items() if k.startswith(prefix)}
    kwargs["marks"] = args.marks 

    name = args.__dict__[component]

    if name not in classes:
        raise ValueError("Unknown {} class {}. Must be one of {}.".format(
            component, name, NAMES[component]))

    component_class = classes[name]
    component_instance = component_class(**kwargs) 

    #print("Instantiated {} of type {}".format(component, name))
    #print("kwargs:")
    #pprint(kwargs)
    #print()

    return component_instance


def get_model(args: Namespace) -> EncDecProcess:
    args.decoder_units_mlp = args.decoder_units_mlp + [args.marks] #[] + [2]

    decoder: Decoder 
    decoder = instantiate_encoder_or_decoder(args, component="decoder") #default = Hawkes

    if decoder.input_size is not None:
        args.encoder_units_mlp = args.encoder_units_mlp + [decoder.input_size]
    #assert((args.encoder_time is None and args.encoder_mark is None and args.encoder is not None) or (args.encoder_time is not None and args.encoder_mark is not None and args.encoder is None ))
    if args.encoder is not None:
        encoder: Encoder
        encoder = instantiate_encoder_or_decoder(args, component="encoder") #default=stub
        process = EncDecProcess(
        encoder=encoder, encoder_time=None, encoder_mark=None, decoder=decoder, multi_labels=args.multi_labels)
    else:
        encoder_time: Encoder
        encoder_mark: Encoder
        encoder_time = instantiate_encoder_or_decoder(args, component="encoder_histtime")
        encoder_mark = instantiate_encoder_or_decoder(args, component="encoder_histmark")
        process = EncDecProcess(
        encoder=None, encoder_time=encoder_time, encoder_mark=encoder_mark, decoder=decoder, multi_labels=args.multi_labels)       
    #Merge encoder and decoder into a single class (subclass of Process, which is a nn.module)
    if args.include_poisson: #True by default
        processes = {process.name: process}
        processes.update({"poisson": PoissonProcess(marks=process.marks)}) #PoissonProcess subclass of EncDecProcess(with Poisson encoder and decoder, i.e. only param by mu)
        process = ModularProcess(
            processes=processes, args=args) #use_coefficients default True
        #merges multiple processes as one process (ex Poisson and stub-Hawkes)
    process = process.to(device=args.device)

    return process
