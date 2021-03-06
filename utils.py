
import torch

import numpy as np
class AvgMeter(object) :
	def __init__(self) :
		self.reset()

	def reset(self) :
		self.sum = 0
		self.count = 0

	def __call__(self, val = None, reset = False) :
		if val is not None :
			self.sum += val
			self.count += 1
		result = 0
		if self.count > 0 :
			result = self.sum / self.count
		if reset :
			self.reset()
		return result

class ImagePool(object) :
	def __init__(self, size, device) :
		self.size = size
		self.bs = 0
		self.count = 0
		self.buffer = None
		self.device = device

	def put(self, images: torch.Tensor) :
		if self.size == 0 :
			return
		if self.bs == 0 :
			self.bs = images.size(0)
			assert self.bs < self.size
		else :
			assert self.bs == images.size(0)
		if self.buffer is None :
			self.buffer = torch.zeros(self.size, *images.shape[1:], dtype = images.dtype, device = self.device)
		remain_cap = self.size - self.count
		if remain_cap >= self.bs :
			# append back
			self.buffer[self.count: self.count + self.bs] = images.detach()
		else :
			self.buffer[remain_cap: remain_cap + self.count] = self.buffer[0: self.count]
			self.buffer[: self.bs] = images.detach()
		self.count = min(self.count + self.bs, self.size)

	def available(self) :
		return self.count > 0

	def sample(self) :
		assert self.count > 0
		indices = list(range(self.count))
		np.random.shuffle(indices)
		indices = indices[: self.bs]
		images = self.buffer[indices].contiguous()
		return images
