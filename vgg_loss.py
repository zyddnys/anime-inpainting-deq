import torch
import torchvision

class VGG16Loss(torch.nn.Module):
	def __init__(self):
		super(VGG16Loss, self).__init__()
		self.vgg = torchvision.models.vgg16(pretrained = True)

	def forward(self, x) :
		for i in range(15) :
			x = self.vgg.features[i](x)
		return x