
import torch
import torch.nn as nn
import random
import os
import numpy as np

from typing import Tuple
from torch.utils.data import Dataset, IterableDataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

import mask

class FileListDataset(IterableDataset) :
	def __init__(self, file_list, image_size = 768, patch_size = 256, root_dir = None) :
		if isinstance(file_list, list) :
			self.samples = file_list
		elif isinstance(file_list, str) :
			with open(file_list, 'r') as fp :
				self.samples = [s.strip() for s in fp.readlines()]
		self.image_size = image_size
		self.patch_size = patch_size
		self.root_dir = root_dir
		self.cache_bg = None
		self.cache_prob = 60 # 60% peob of using cache

	@staticmethod
	def resize_keep_aspect(img, size) :
		ratio = (float(size)/min(img.size[0], img.size[1]))
		new_width = round(img.size[0] * ratio)
		new_height = round(img.size[1] * ratio)
		return img.resize((new_width, new_height), Image.ANTIALIAS)

	@staticmethod
	def read_image_file(filename) :
		img = Image.open(filename)
		if img.mode == 'RGBA' :
			# from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
			img.load()  # needed for split()
			background = Image.new('RGB', img.size, (255, 255, 255))
			background.paste(img, mask = img.split()[3])  # 3 is the alpha channel
			return background
		elif img.mode == 'P' :
			img = img.convert('RGBA')
			img.load()  # needed for split()
			background = Image.new('RGB', img.size, (255, 255, 255))
			background.paste(img, mask = img.split()[3])  # 3 is the alpha channel
			return background
		else :
			return img.convert('RGB')

	def __next__(self) -> Tuple[torch.Tensor, torch.Tensor] :
		if self.cache_bg is None or np.random.randint(0, 100) > self.cache_prob :
			img_filename = np.random.choice(self.samples)
			if self.root_dir :
				img = self.read_image_file(os.path.join(self.root_dir, img_filename))
			else :
				img = self.read_image_file(img_filename)
			img = self.resize_keep_aspect(img, self.image_size)
			self.cache_bg = img
		else :
			img = self.cache_bg
		patch = transforms.RandomCrop(self.patch_size, fill = (255, 255, 255), pad_if_needed = True)(img)
		patch_img = F.to_tensor(patch) * 2. - 1.
		mask_img = mask.mask_image(patch_img, self.patch_size, self.patch_size)
		return patch_img, mask_img

	def __iter__(self) :
		return self

def init_worker(seed) :
	seed = (seed + np.random.randint(0, 114514)) & 0xffffffff
	for i in range(10):
		seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
	np.random.seed(seed)
	random.seed(seed)

def test() :
	import time, sys
	ds = FileListDataset('train.flist')
	loader = torch.utils.data.DataLoader(
		ds,
		batch_size = 8,
		num_workers = 16,
		worker_init_fn = init_worker,
		pin_memory = True
	)
	start = time.time()
	for img, mask in loader :
		end = time.time()
		print('%.2fms' % ((end - start) * 1000), file=sys.stderr)
		start = time.time()

if __name__ == '__main__' :
	test()
