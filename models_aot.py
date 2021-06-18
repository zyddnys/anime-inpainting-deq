
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
from torch.nn.utils import spectral_norm
class Discriminator(nn.Module) :
	def __init__(self, in_ch = 3, in_planes = 64, blocks = [2, 2, 2], alpha = 0.2) :
		super(Discriminator, self).__init__()
		self.in_planes = in_planes

		self.conv = nn.Sequential(
			spectral_norm(nn.Conv2d(in_ch, in_planes, 4, stride=2, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(in_planes, in_planes*2, 4, stride=2, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(in_planes*2, in_planes*4, 4, stride=2, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
			spectral_norm(nn.Conv2d(in_planes*4, in_planes*8, 4, stride=1, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(512, 1, 4, stride=1, padding=1)
		)

	def forward(self, x) :
		x = self.conv(x)
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

		self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(10)])

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
