
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.vgg import vgg16

import ganloss
import utils
import dataset
import models_vanilla as models
import vgg_loss

from tqdm import tqdm

def mask_image(img, mask) :
	return img * (1. - mask)

def img_unscale(img) :
	return (img.detach() + 1) * .5

def train(
	network_gen: nn.Module,
	network_dis: nn.Module,
	dataloader,
	checkpoint_path,
	weight_gan = 0.05,
	weight_l1 = 1.0,
	weight_fm = 0.5,
	device = torch.device('cuda:0'),
	n_critic = 1,
	n_gen = 1,
	fake_pool_size = 256,
	lr_gen = 1e-4,
	lr_dis = 5e-4,
	updates_per_epoch = 10000,
	record_freq = 1000,
	total_updates = 1000000,
	gradient_accumulate = 4,
	enable_fp16 = True,
	resume = False
	) :
	print(' -- Initializing losses')
	loss_gan = ganloss.GANLossHinge(device)
	loss_vgg = vgg_loss.VGG16Loss().to(device)

	opt_gen = optim.AdamW(network_gen.parameters(), lr = lr_gen, betas = (0.5, 0.99), weight_decay = 1e-6)
	opt_dis = optim.AdamW(network_dis.parameters(), lr = lr_dis, betas = (0.5, 0.99), weight_decay = 1e-6)

	sch_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, 'min', factor = 0.5, patience = 4, verbose = True, min_lr = 1e-6)
	sch_dis = optim.lr_scheduler.ReduceLROnPlateau(opt_dis, 'min', factor = 0.5, patience = 4, verbose = True, min_lr = 1e-6)
	sch_meter = utils.AvgMeter()
	
	scaler_gen = amp.GradScaler(enabled = enable_fp16)
	scaler_dis = amp.GradScaler(enabled = enable_fp16)

	loss_dis_real_meter = utils.AvgMeter()
	loss_dis_fake_meter = utils.AvgMeter()
	loss_dis_meter = utils.AvgMeter()

	loss_gen_l1_meter = utils.AvgMeter()
	loss_gen_l1_coarse_meter = utils.AvgMeter()
	loss_gen_fm_meter = utils.AvgMeter()
	loss_gen_gan_meter = utils.AvgMeter()
	loss_gen_meter = utils.AvgMeter()

	writer = SummaryWriter(os.path.join(checkpoint_path, 'tb_summary'))
	os.makedirs(os.path.join(checkpoint_path, 'checkpoints'), exist_ok = True)

	fakepool = utils.ImagePool(fake_pool_size, device)

	counter_start = 0
	if resume :
		chekcpoints = os.listdir(os.path.join(checkpoint_path, 'checkpoints'))
		last_chekcpoints = sorted(chekcpoints, key = lambda item: (len(item), item))[-1] if 'latest.ckpt' not in chekcpoints else 'latest.ckpt'
		print(f' -- Loading checkpoint {last_chekcpoints}')
		ckpt = torch.load(os.path.join(checkpoint_path, 'checkpoints', last_chekcpoints))
		network_gen.load_state_dict(ckpt['gen'])
		network_dis.load_state_dict(ckpt['dis'])
		opt_gen.load_state_dict(ckpt['gen_opt'])
		opt_dis.load_state_dict(ckpt['dis_opt'])
		counter_start = ckpt['counter'] + 1
		print(f' -- Resume training from update {counter_start}')
	else :
		print(f' -- Start training from scratch')

	dataloader = iter(dataloader)

	print(' -- Training start')
	try :
		for counter in tqdm(range(counter_start, total_updates)) :
			# train discrimiantor
			for critic in range(n_critic) :
				opt_dis.zero_grad()
				for _ in range(gradient_accumulate) :
					real_img, mask = next(dataloader)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					if np.random.randint(0, 2) == 0 or not fakepool.available() :
						with torch.no_grad(), amp.autocast(enabled = enable_fp16) :
							_, fake_img = network_gen(real_img_masked, mask)
						fakepool.put(fake_img)
					else :
						fake_img = fakepool.sample()
					with amp.autocast(enabled = enable_fp16) :
						real_logits = network_dis(real_img)
						fake_logits = network_dis(fake_img)
						loss_dis_real = loss_gan(real_logits, 'real')
						loss_dis_fake = loss_gan(fake_logits, 'fake')
						loss_dis = 0.5 * (loss_dis_real + loss_dis_fake)
					if torch.isnan(loss_dis) or torch.isinf(loss_dis) :
						breakpoint()

					scaler_dis.scale(loss_dis / float(gradient_accumulate)).backward()

					loss_dis_real_meter(loss_dis_real.item())
					loss_dis_fake_meter(loss_dis_fake.item())
					loss_dis_meter(loss_dis.item())
				scaler_dis.unscale_(opt_dis)
				scaler_dis.step(opt_dis)
				scaler_dis.update()
			# train generator
			for gen in range(n_gen) :
				opt_gen.zero_grad()
				for _ in range(gradient_accumulate) :
					real_img, mask = next(dataloader)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					with amp.autocast(enabled = enable_fp16) :
						inpainted_result_coarse, inpainted_result = network_gen(real_img_masked, mask)
						with torch.no_grad() :
							real_image_feats = loss_vgg(real_img)
						fake_image_feats = loss_vgg(inpainted_result)
						loss_gen_l1 = F.l1_loss(inpainted_result, real_img)
						loss_gen_l1_coarse = F.l1_loss(inpainted_result_coarse, real_img)
						loss_gen_fm = F.l1_loss(fake_image_feats, real_image_feats)
						generator_logits = network_dis(inpainted_result)
						loss_gen_gan = loss_gan(generator_logits, 'generator')
						loss_gen = weight_l1 * (loss_gen_l1 + loss_gen_l1_coarse) + weight_fm * loss_gen_fm + weight_gan * loss_gen_gan
					if torch.isnan(loss_gen) or torch.isinf(loss_dis) :
						breakpoint()

					scaler_gen.scale(loss_gen / float(gradient_accumulate)).backward()

					loss_gen_meter(loss_gen.item())
					loss_gen_l1_meter(loss_gen_l1.item())
					loss_gen_l1_coarse_meter(loss_gen_l1_coarse.item())
					sch_meter(loss_gen_l1.item()) # use L1 loss as lr scheduler metric
					loss_gen_fm_meter(loss_gen_fm.item())
					loss_gen_gan_meter(loss_gen_gan.item())
				scaler_gen.unscale_(opt_gen)
				scaler_gen.step(opt_gen)
				scaler_gen.update()
			if counter % record_freq == 0 :
				tqdm.write(f' -- Record at update {counter}')
				writer.add_scalar('discriminator/all', loss_dis_meter(reset = True), counter)
				writer.add_scalar('discriminator/real', loss_dis_real_meter(reset = True), counter)
				writer.add_scalar('discriminator/fake', loss_dis_fake_meter(reset = True), counter)
				writer.add_scalar('generator/all', loss_gen_meter(reset = True), counter)
				writer.add_scalar('generator/l1', loss_gen_l1_meter(reset = True), counter)
				writer.add_scalar('generator/l1_coarse', loss_gen_l1_coarse_meter(reset = True), counter)
				writer.add_scalar('generator/fm', loss_gen_fm_meter(reset = True), counter)
				writer.add_scalar('generator/gan', loss_gen_gan_meter(reset = True), counter)
				writer.add_image('original/image', img_unscale(real_img), counter, dataformats = 'NCHW')
				writer.add_image('original/mask', mask, counter, dataformats = 'NCHW')
				writer.add_image('original/masked', img_unscale(real_img_masked), counter, dataformats = 'NCHW')
				writer.add_image('inpainted/refined', img_unscale(inpainted_result), counter, dataformats = 'NCHW')
				writer.add_image('inpainted/coarse', img_unscale(inpainted_result_coarse), counter, dataformats = 'NCHW')
				torch.save(
					{
						'dis': network_dis.state_dict(),
						'gen': network_gen.state_dict(),
						'dis_opt': opt_dis.state_dict(),
						'gen_opt': opt_gen.state_dict(),
						'counter': counter
					},
					os.path.join(checkpoint_path, 'checkpoints', f'update_{counter}.ckpt')
				)
			if counter > 0 and counter % updates_per_epoch == 0 :
				tqdm.write(f' -- Epoch finished at update {counter}')
				# epoch finished
				loss_epoch = sch_meter(reset = True)
				sch_gen.step(loss_epoch)
				sch_dis.step(loss_epoch)
	except KeyboardInterrupt :
		print(' -- Training interrupted, saving latest model ..')
		torch.save(
			{
				'dis': network_dis.state_dict(),
				'gen': network_gen.state_dict(),
				'dis_opt': opt_dis.state_dict(),
				'gen_opt': opt_gen.state_dict(),
				'counter': counter
			},
			os.path.join(checkpoint_path, 'checkpoints', f'latest.ckpt')
		)

def main(args, device) :
	print(' -- Initializing models')
	gen = models.InpaintingVanilla().to(device)
	dis = models.DiscriminatorSimple().to(device)
	ds = dataset.FileListDataset('train.flist', image_size_min = args.image_file_size_min, image_size_max = args.image_file_size_max, patch_size = args.image_size)
	loader = torch.utils.data.DataLoader(
		ds,
		batch_size = args.batch_size,
		num_workers = args.workers,
		worker_init_fn = dataset.init_worker,
		pin_memory = True
	)
	train(gen, dis, loader, args.checkpoint_dir, gradient_accumulate = args.gradient_accumulate, resume = args.resume)

if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-dir', '-d', type = str, default = './checkpoints', help = "where to place checkpoints")
	parser.add_argument('--batch-size', type = int, default = 4, help = "training batch size")
	parser.add_argument('--resume', action = 'store_true', help = "resume training")
	parser.add_argument('--enable-amp', action = 'store_false', help = "enable amp fp16 training")
	parser.add_argument('--enable-tf32', action = 'store_true', help = "enable tf32 training for NVIDIA Ampere GPU")
	parser.add_argument('--gradient-accumulate', type = int, default = 8, help = "gradient accumulate")
	parser.add_argument('--image-size', type = int, default = 320, help = "size of cropped patch used for training")
	parser.add_argument('--image-file-size-min', type = int, default = 640, help = "lower bound of smallest axis of image before cropping")
	parser.add_argument('--image-file-size-max', type = int, default = 1920, help = "upper bound of smallest axis of image before cropping")
	parser.add_argument('--workers', type = int, default = 24, help = "num of dataloader workers")
	args = parser.parse_args()
	if args.enable_tf32 :
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	main(args, torch.device("cuda:0"))
