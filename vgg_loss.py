import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import einops

class VGG16Loss(torch.nn.Module):
	def __init__(self):
		super(VGG16Loss, self).__init__()
		self.vgg = torchvision.models.vgg16(pretrained = True)

	def forward(self, x) :
		for i in range(15) :
			x = self.vgg.features[i](x)
		return x

# from https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/src/loss/loss.py
class VGG19(nn.Module):
	def __init__(self, resize_input=False):
		super(VGG19, self).__init__()
		features = torchvision.models.vgg19(pretrained=True).features

		self.resize_input = resize_input
		self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]))
		self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]))
		prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
		posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
		names = list(zip(prefix, posfix))
		self.relus = []
		for pre, pos in names:
			self.relus.append('relu{}_{}'.format(pre, pos))
			self.__setattr__('relu{}_{}'.format(
				pre, pos), torch.nn.Sequential())

		nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
				[9, 10, 11], [12, 13], [14, 15], [16, 17],
				[18, 19, 20], [21, 22], [23, 24], [25, 26],
				[27, 28, 29], [30, 31], [32, 33], [34, 35]]

		for i, layer in enumerate(self.relus):
			for num in nums[i]:
				self.__getattr__(layer).add_module(str(num), features[num])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		# resize and normalize input for pretrained vgg19
		x = (x + 1.0) / 2.0
		x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
		if self.resize_input:
			x = F.interpolate(
				x, size=(256, 256), mode='bilinear', align_corners=True)
		features = []
		for layer in self.relus:
			x = self.__getattr__(layer)(x)
			features.append(x)
		out = {key: value for (key, value) in list(zip(self.relus, features))}
		return out

def gram_matrix(y) :
	b, ch, h, w = y.shape
	return torch.einsum('bchw,bdhw->bcd', [y, y]) / (h * w)
	
class VGG19LossWithStyle(torch.nn.Module):
	def __init__(self, w_fm = 0.1, w_style = 250):
		super(VGG19LossWithStyle, self).__init__()
		self.vgg = VGG19()
		self.fm_vgg_stages = [2, 3, 4, 5]
		self.style_prefix = [2, 3, 4, 5]
		self.style_postfix = [2, 4, 4, 2]
		self.w_fm = w_fm
		self.w_style = w_style

	def forward(self, inp, target) :
		inp_vgg = self.vgg(inp)
		with torch.no_grad() :
			target_vgg = self.vgg(target)
		loss_fm = 0.0
		loss_style = 0.0
		for i in range(4) :
			loss_fm += F.l1_loss(inp_vgg[f'relu{self.fm_vgg_stages[i]}_1'], target_vgg[f'relu{self.fm_vgg_stages[i]}_1'])
			loss_style += F.l1_loss(gram_matrix(inp_vgg[f'relu{self.style_prefix[i]}_{self.style_postfix[i]}']),
				gram_matrix(target_vgg[f'relu{self.style_prefix[i]}_{self.style_postfix[i]}']))
		return self.w_fm * loss_fm + self.w_style * loss_style
			
def main() :
	img = torch.randn(4, 3, 256, 256)
	img_ref = torch.randn(4, 3, 256, 256)
	l = VGG19LossWithStyle(1, 1)
	loss = l(img, img_ref)
	print(loss)

if __name__ == '__main__' :
	main()
