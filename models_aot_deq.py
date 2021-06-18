
from typing import Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from mdeq_forward_backward import MDEQWrapper

def relu_nf(x) :
	return F.relu(x) * 1.7139588594436646

def gelu_nf(x) :
	return F.gelu(x) * 1.7015043497085571

def silu_nf(x) :
	return F.silu(x) * 1.7881293296813965

def sigmoid_nf(x) :
	return F.sigmoid(x) * 4.803835391998291

class LambdaLayer(nn.Module) :
	def __init__(self, f):
		super(LambdaLayer, self).__init__()
		self.f = f

	def forward(self, x) :
		return self.f(x)

def my_layer_norm(feat):
	mean = feat.mean((2, 3), keepdim=True)
	std = feat.std((2, 3), keepdim=True) + 1e-9
	feat = 2 * (feat - mean) / std
	feat = 5 * feat
	return feat


class ScaledWSConv2d(nn.Conv2d):
	"""2D Conv layer with Scaled Weight Standardization."""
	def __init__(self, in_channels, out_channels, kernel_size,
				stride=1, padding=0,
				dilation=1, groups=1, bias=True, gain=True,
				eps=1e-4):
		nn.Conv2d.__init__(self, in_channels, out_channels,
				kernel_size, stride,
				padding, dilation,
				groups, bias)
		#nn.init.kaiming_normal_(self.weight)
		if gain:
			self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
		else:
			self.gain = None
		# Epsilon, a small constant to avoid dividing by zero.
		self.eps = eps

	def get_weight(self):
		# Get Scaled WS weight OIHW;
		fan_in = np.prod(self.weight.shape[1:])
		mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
		var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
		weight = (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
		if self.gain is not None:
			weight = weight * self.gain
		return weight

	def forward(self, x):
		return F.conv2d(x, self.get_weight(), self.bias,
			self.stride, self.padding,
			self.dilation, self.groups)

class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
	"""2D Transpose Conv layer with Scaled Weight Standardization."""
	def __init__(self, in_channels: int,
		out_channels: int,
		kernel_size,
		stride = 1,
		padding = 0,
		output_padding = 0,
		groups: int = 1,
		bias: bool = True,
		dilation: int = 1,
		gain=True,
		eps=1e-4):
		nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros')
		#nn.init.kaiming_normal_(self.weight)
		if gain:
			self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
		else:
			self.gain = None
		# Epsilon, a small constant to avoid dividing by zero.
		self.eps = eps
	def get_weight(self):
		# Get Scaled WS weight OIHW;
		fan_in = np.prod(self.weight.shape[1:])
		var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
		scale = torch.rsqrt(torch.max(
			var * fan_in, torch.tensor(self.eps).to(var.device))) * self.gain.view_as(var).to(var.device)
		shift = mean * scale
		return self.weight * scale - shift
		
	def forward(self, x, output_size: Optional[List[int]] = None):
		output_padding = self._output_padding(
			input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
		return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride, self.padding,
			output_padding, self.groups, self.dilation)

class AOTBlockNF(nn.Module) :
	def __init__(self, dim, rates = [2, 4, 8, 16]):
		super(AOTBlockNF, self).__init__()
		self.rates = rates
		for i, rate in enumerate(rates):
			self.__setattr__(
				'block{}'.format(str(i).zfill(2)), 
				nn.Sequential(
					nn.ReflectionPad2d(rate),
					ScaledWSConv2d(dim, dim // 4, 3, padding = 0, dilation = rate),
					LambdaLayer(gelu_nf)
				)
			)
		self.fuse = nn.Sequential(
			nn.ReflectionPad2d(1),
			ScaledWSConv2d(dim, dim, 3, padding = 0)
		)
		self.gate = nn.Sequential(
			nn.ReflectionPad2d(1),
			ScaledWSConv2d(dim, dim, 3, padding = 0)
		)

	def forward(self, x):
		out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
		out = torch.cat(out, 1)
		out = self.fuse(out)
		mask = my_layer_norm(self.gate(x))
		mask = torch.sigmoid(mask)
		nf_out = (x * (1 - mask) + out * mask)
		nf_out = nf_out * 1.09
		return nf_out

class AOTBlockInjectionNF(nn.Module) :
	def __init__(self, dim, rates = [2, 4, 8, 16]):
		super(AOTBlockInjectionNF, self).__init__()
		self.rates = rates
		for i, rate in enumerate(rates):
			self.__setattr__(
				'block{}'.format(str(i).zfill(2)), 
				nn.Sequential(
					nn.ReflectionPad2d(rate),
					ScaledWSConv2d(dim, dim // 4, 3, padding = 0, dilation = rate),
					LambdaLayer(gelu_nf)
				)
			)
		self.fuse = nn.Sequential(
			nn.ReflectionPad2d(1),
			ScaledWSConv2d(dim, dim, 3, padding = 0)
		)
		self.gate = nn.Sequential(
			nn.ReflectionPad2d(1),
			ScaledWSConv2d(dim, dim, 3, padding = 0)
		)
		self.x_conv = ScaledWSConv2d(dim, dim, 1)
		self.z_conv = ScaledWSConv2d(dim, dim, 1)
		self.gate_conv = ScaledWSConv2d(dim, dim, 1)

	def forward(self, x, injection):
		if injection is not None :
			gate = self.gate_conv(self.z_conv(x) + self.x_conv(injection)).sigmoid()
			x = x * gate + (1 - gate) * injection
		out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
		out = torch.cat(out, 1)
		out = self.fuse(out)
		mask = my_layer_norm(self.gate(x))
		mask = torch.sigmoid(mask)
		nf_out = (x * (1 - mask) + out * mask)
		nf_out = nf_out * 1.09
		return nf_out

class ModelF(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelF, self).__init__()
		self.cfg = cfg
		self.inj = AOTBlockInjectionNF(cfg.get('planes', 512))
		self.c1 = AOTBlockNF(cfg.get('planes', 512))
		self.c2 = AOTBlockNF(cfg.get('planes', 512))

	def forward(self, z_list: List[torch.Tensor], x_list: List[torch.Tensor], *args) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of function f, z[i+1]=f(z[i],x)
		length of z_list and x_list must be the same
		shape of tensors in z_list and x_list must be the same
		DO NOT use BatchNorm or anything that can leak information between samples here
		params:
			z_list: list of z tensors, the fixed point to be solved
			x_list: list of x tensors, input injection
		returns:
			list of z tensors, must be the same size and shape of the input z tensors
		"""
		h = [self.inj(z, x) for z, x in zip(z_list, x_list)]
		h = [self.c1(z) for z in h]
		h = [self.c2(z) for z in h]
		return h

	def init_z_list(self, x_list: List[torch.Tensor]) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of initializing z tensor list, usually zeros tensors
		Notice here x input injection is provided as a parameter so you can initialize z with the correct shape on the correct device
		params:
			x_list: list of x tensors, input injection
		returns:
			list of initial z tensors
		"""
		return [torch.zeros_like(x) for x in x_list]

class ModelInjection(nn.Module) :
	def __init__(self, cfg: dict[str: Any]):
		super(ModelInjection, self).__init__()
		self.cfg = cfg
		self.pad = nn.ReflectionPad2d(1)
		self.stem = ScaledWSConv2d(3, 64, 3, 1, 0)
		self.l1 = ScaledWSConv2d(64, 128, 4, 2, 0)
		self.l2 = ScaledWSConv2d(128, 256, 4, 2, 0)
		self.proj = ScaledWSConv2d(256, cfg.get('planes', 512), 1)

	def forward(self, x: torch.Tensor) -> List[torch.Tensor] :
		"""
		Fill here with your implementation of input injection
		This module takes your input and extend it to a list of input injections for all scales
		(Notice you can change the input of this module to whatever you need)
		You can use BatchNorm here
		params:
			x: your single input
		returns:
			list of initial x tensors, input injection
		"""
		h = gelu_nf(self.stem(self.pad(x)))
		h = gelu_nf(self.l1(self.pad(h)))
		h = gelu_nf(self.l2(self.pad(h)))
		h = gelu_nf(self.proj(h))
		return [h]

class MDEQModelBackbone(nn.Module) :
	def __init__(self, cfg: dict[str: Any], f: ModelF, inject: ModelInjection):
		super(MDEQModelBackbone, self).__init__()
		self.parse_cfg(cfg)
		self.f = f
		self.inject = inject
		self.f_copy = copy.deepcopy(self.f)
			
		for param in self.f_copy.parameters():
			param.requires_grad_(False)
		self.deq = MDEQWrapper(self.f, self.f_copy)

	def parse_cfg(self, cfg: dict[str: Any]):
		self.num_layers = cfg.get('num_layers', 2)
		self.f_thres = cfg.get('f_thres', 24)
		self.b_thres = cfg.get('b_thres', 24)
		self.pretrain_steps = cfg.get('pretrain_steps', 0)

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) -> List[torch.Tensor] :
		f_thres = kwargs.get('f_thres', self.f_thres)
		b_thres = kwargs.get('b_thres', self.b_thres)
		writer = kwargs.get('writer', None)     # For tensorboard
		self.f_copy.load_state_dict(self.f.state_dict())
		x_list = self.inject(x)
		z_list = self.f.init_z_list(x_list)
		if 0 <= train_step < self.pretrain_steps:
			for layer_ind in range(self.num_layers):
				z_list = self.f(z_list, x_list)
		else:
			if train_step == self.pretrain_steps:
				torch.cuda.empty_cache()
				print(' -- Switching to DEQ')
			z_list = self.deq(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)
		return z_list

class MDEQModelAOT(MDEQModelBackbone) :
	def __init__(self, cfg: dict[str: Any]):
		super(MDEQModelAOT, self).__init__(cfg, ModelF(cfg), ModelInjection(cfg))
		self.pad = nn.ReflectionPad2d(1)
		self.proj = ScaledWSConv2d(cfg.get('planes', 512), 256, 1)
		self.deconv1 = ScaledWSTransposeConv2d(256, 128, 4, 2, 1)
		self.deconv2 = ScaledWSTransposeConv2d(128, 64, 4, 2, 1)
		self.to_rgb = ScaledWSConv2d(64, 3, 3, 1, 0)

	def forward(self, x: torch.Tensor, train_step = -1, **kwargs) :
		"""
		Fill here with your implementation of model head
		(Usually a classification head)
		params:
			x: your single input
		returns:
			Whatever your output want to be
		"""
		h: List[torch.Tensor] = super().forward(x, train_step, **kwargs)
		h = h[0]
		h = gelu_nf(self.proj(h))
		h = gelu_nf(self.deconv1(h))
		h = gelu_nf(self.deconv2(h))
		img = gelu_nf(self.to_rgb(self.pad(h)))
		return img


def test() :
	cfg = {
		'num_layers': 4,
		'f_thres': 22,
		'b_thres': 24,
		'pretrain_steps' : 10000,
		'planes': 512
	}
	m = MDEQModelAOT(cfg).cuda()
	img = torch.randn(2, 3, 512, 512).cuda()
	x = m(img)
	target = torch.randn(2, 3, 512, 512).cuda()
	F.l1_loss(x, target).backward()

if __name__ == '__main__' :
	test()
