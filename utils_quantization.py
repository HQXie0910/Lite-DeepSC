
"""
This is the until for quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""
1. find the best min and max value
2. use the min and max to caculate the scale and zero point
3. use the scale and zero point to quantize weights and activation
4. quantization-aware training and calibration on activation for better performance
"""

"""
********************* range_trackers(范围统计器，统计量化前范围) *********************
1. MinMax range trackers  ---> Global Range Tracker
2. Moving Average MinMax trackers ---> Averaged Range Tracker
3. KL divergence MinMax trackers (unfinished)
"""
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':    
            # A, min_max_shape=(1, 1, 1, 1),layer级
            min_val = torch.min(input) + 1e-4
            max_val = torch.max(input) + 1e-4
        elif self.q_level == 'C':  
            # W, min_max_shape=(N, 1, 1, 1),channel级
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            
        self.update_range(min_val, max_val)
        
class AveragedRangeTracker(RangeTracker):  
    # A, min_max_shape=(1, 1, 1, 1), layer级, 取running_min_max —— (N, C, W, H)
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            # min^(n) = (1 - gamma)min^(n-1) + gamma*x_min
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum) 
            # max^(n) = (1 - gamma)max^(n-1) + gamma*x_max
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)
            
class GlobalRangeTracker(RangeTracker):  
    # W, min_max_shape=(N, 1, 1, 1), channel级, 取本次和之前相比的min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            # x_min = min(x_min, min(X))
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            # x_max = max(x_max, max(X))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))
            

class KLRangeTracker(nn.Module):
    def __init__(self, q_level):
        super(KLRangeTracker).__init__()
        self.q_level = q_level
"""
## ********************* quantizers *********************
1. straight-through_estimator bridge the backward --> Round
2. Symmetric Quantizer
3. Asymmetric Quantizer
"""

class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', torch.tensor(0))      # 量化比例因子
        self.register_buffer('zero_point', torch.tensor(0)) # 量化零点

    def update_params(self):
        raise NotImplementedError

    # 量化
    def quantize(self, input):
        output = input * self.scale - self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 截断
    def clamp(self, input):
        """
              | min, if x_i < min
        y_i = | x_i, if min <= x_i <= max
              | max, if x_i > max
        """
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    # 反量化
    def dequantize(self, input):
        output = (input + self.zero_point) / self.scale 
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            self.range_tracker(input)       # caculate max and min value
            self.update_params()            # caculate scale and zero point value
            output = self.quantize(input)   #
            output = self.round(output)     #
            output = self.clamp(output)     #
            output = self.dequantize(output)#
        return output
    
class SignedQuantizer(Quantizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the range of sign number is (-2^(bits - 1), 2^(bits - 1) - 1 )
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))    # -2^(bits - 1)
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1)) # 2^(bits - 1) - 1 
        
class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the range of unsign number is (0, 2^bits  - 1 )
        self.register_buffer('min_val', torch.tensor(0))                    # 0
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1)) # 2^bits  - 1 

class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        """
        scale = 2^(bits - 1)/max(x_float)
        zero_point = 0
        """
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), 
                                torch.abs(self.range_tracker.max_val))  #
        self.scale = quantized_range / float_range
        self.zero_point = torch.zeros_like(self.scale)

class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        """
        scale = 2^(bits - 1)/(max(x_float) - min(x_float))
        zero_point = min(x_float)*scale
        """
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = quantized_range / (float_range + 1e-8)
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)

"""
1. Quantized Linear layer
2. Quantized Conv2d layer
"""

class QuantizedLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
    super().__init__(
            in_features = in_features, 
            out_features = out_features, 
            bias = bias)
    
    self.activation_quantizer = AsymmetricQuantizer(bits = a_bits, 
                                                    range_tracker = AveragedRangeTracker(q_level='L'))
    self.weight_quantizer = AsymmetricQuantizer(bits = w_bits, 
                                                range_tracker = AveragedRangeTracker (q_level='L'))

  def forward(self, input):
    q_input = self.activation_quantizer(input)
    q_weight = self.weight_quantizer(self.weight)
    output = F.linear(input=q_input, weight=q_weight, bias=self.bias)

    
    return output

class QuantizedLinear_cons(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
    super().__init__(
            in_features = in_features, 
            out_features = out_features, 
            bias = bias)
    
    self.activation_quantizer = AsymmetricQuantizer(bits = a_bits, 
                                                    range_tracker = AveragedRangeTracker(q_level='L'))
    self.weight_quantizer = AsymmetricQuantizer(bits = w_bits, 
                                                range_tracker = AveragedRangeTracker (q_level='L'))

  def forward(self, input):
    q_weight = self.weight_quantizer(self.weight)
    output = F.linear(input=input, weight=q_weight, bias=self.bias)
    return output

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups = 1, 
                 bias = True, 
                 a_bits = 8, 
                 w_bits = 8, 
                 q_type = 1, 
                 first_layer = 0, ):
        super().__init__( 
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding,
                dilation = dilation,
                groups = groups, 
                bias = bias
                )
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, 
                                                           range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, 
                                                       range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, 
                                                            range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, 
                                                        range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight) 
        # 量化卷积
        output = F.conv2d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output

def reshape_to_activation(input):
  return input.reshape(1, -1, 1, 1)
def reshape_to_weight(input):
  return input.reshape(-1, 1, 1, 1)
def reshape_to_bias(input):
  return input.reshape(-1)
