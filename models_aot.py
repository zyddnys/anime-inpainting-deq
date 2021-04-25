
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules import *

class ResBlockDis(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(ResBlockDis, self).__init__()
		self.bn1 = nn.InstanceNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3 if stride == 1 else 4, stride=stride, padding=1)
		self.bn2 = nn.InstanceNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
		self.planes = planes
		self.in_planes = in_planes
		self.stride = stride

		self.shortcut = nn.Sequential()
		if stride > 1 :
			self.shortcut = nn.Sequential(nn.AvgPool2d(2, 2), nn.Conv2d(in_planes, planes, kernel_size=1))
		elif in_planes != planes and stride == 1 :
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1))

	def forward(self, x):
		sc = self.shortcut(x)
		x = self.conv1(F.leaky_relu(self.bn1(x), 0.2))
		x = self.conv2(F.leaky_relu(self.bn2(x), 0.2))
		return sc + x

class Discriminator(nn.Module) :
	def __init__(self, in_ch = 3, in_planes = 64, blocks = [2, 2, 2], alpha = 0.2) :
		super(Discriminator, self).__init__()
		self.in_planes = in_planes

		self.conv1 = ScaledWSConv2d(in_ch, in_planes, kernel_size=4, stride=2, padding=1, bias=True)
		self.layers = []
		planes = self.in_planes
		for nb in blocks :
			self.layers.append(self._make_layer(ResBlockDis, planes, nb, stride=2))
			planes *= 2
		self.layers = nn.Sequential(*self.layers)
		self.cls = ScaledWSConv2d(planes // 2, 1, 1)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for i, stride in enumerate(strides) :
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes

		return nn.Sequential(*layers)

	def forward(self, x) :
		x = self.conv1(x)
		for l in self.layers :
			x = l(x)
		x = self.cls(relu_nf(x))
		return x

class AOTGenerator(nn.Module) :
	def __init__(self, in_ch = 4, out_ch = 3, ch = 32, alpha = 0.0) :
		super(AOTGenerator, self).__init__()

		self.head = nn.Sequential(
			GatedWSConvPadded(in_ch, ch, 3, stride = 1),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, ch * 2, 4, stride = 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 2, ch * 4, 4, stride = 2),
		)

		self.beta = 1.0
		self.alpha = alpha
		self.body_conv = []
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(AOTBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv.append(ResBlock(ch * 4, self.alpha, self.beta))
		self.beta = (self.beta ** 2 + self.alpha ** 2) ** 0.5
		self.body_conv = nn.Sequential(*self.body_conv)

		self.tail = nn.Sequential(
			GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
			LambdaLayer(relu_nf),
			GatedWSConvPadded(ch, out_ch, 3, stride = 1),
		)

	def forward(self, img, mask) :
		x = torch.cat([mask, img], dim = 1)
		x = self.head(x)
		conv = self.body_conv(x)
		x = self.tail(conv)
		if self.training :
			return x
		else :
			return torch.clip(x, -1, 1)

def test() :
	img = torch.randn(4, 3, 256, 256).cuda()
	mask = torch.randn(4, 1, 256, 256).cuda()
	net = AOTGenerator().cuda()
	y1 = net(img, mask)
	print(y1.shape)


if __name__ == '__main__' :
	test()
