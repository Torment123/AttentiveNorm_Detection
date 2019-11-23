from .hrnet import HRNet
from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .resnet_ivmcl import ResNet_iVMCL
#from .aognet import AOGNet
from .aognet.aognet import AOGNet

__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet', 'ResNet_iVMCL', 'AOGNet']
