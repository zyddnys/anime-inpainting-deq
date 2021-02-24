
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import ganloss
import utils

def mask_image(img, mask) :
	return img * mask

def img_unscale(img) :
	return (img.detach() + 1) * .5

def train(
	network_gen: nn.Module,
	network_dis: nn.Module,
	dataloader,
	checkpoint_path,
	weight_gan = 0.01,
	weight_l1 = 1.0,
	weight_fm = 1.0,
	device = torch.device('cuda:0'),
	n_critic = 3,
	n_gen = 1,
	fake_pool_size = 0,
	lr_gen = 1e-4,
	lr_dis = 5e-4,
	updates_per_epoch = 10000,
	record_freq = 1000,
	total_updates = 1000000,
	gradient_accumulate = 8,
	enable_fp16 = True
	) :
	loss_gan = ganloss.GANLossSCE(device)

	opt_gen = optim.AdamW(network_gen.parameters(), lr = lr_gen, betas = (0.5, 0.99))
	opt_dis = optim.AdamW(network_gen.parameters(), lr = lr_dis, betas = (0.5, 0.99))

	sch_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, 'min', factor = 0.5, patience = 4, verbose = True, min_lr = 1e-6)
	sch_dis = optim.lr_scheduler.ReduceLROnPlateau(opt_dis, 'min', factor = 0.5, patience = 4, verbose = True, min_lr = 1e-6)
	sch_meter = utils.AvgMeter()
	
	scaler = amp.GradScaler(enabled = enable_fp16)

	loss_dis_real_meter = utils.AvgMeter()
	loss_dis_fake_meter = utils.AvgMeter()
	loss_dis_meter = utils.AvgMeter()

	loss_gen_l1_meter = utils.AvgMeter()
	loss_gen_fm_meter = utils.AvgMeter()
	loss_gen_gan_meter = utils.AvgMeter()
	loss_gen_meter = utils.AvgMeter()

	writer = SummaryWriter(os.path.join(checkpoint_path, 'tb_summary'))
	os.makedirs(os.path.join(checkpoint_path, 'checkpoints'), exist_ok = True)

	# TODO: restore from checkpoint if present

	try :
		for counter in range(total_updates) :
			# train discrimiantor
			for critic in range(n_critic) :
				opt_dis.zero_grad()
				for _ in range(gradient_accumulate) :
					real_img, mask = next(dataloader)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					if np.random.randint(0, 2) == 0 :
						with torch.no_grad(), amp.autocast(enabled = enable_fp16) :
							fake_img = network_gen(real_img_masked, mask)
						# TODO: add to fake pool
					else :
						# TODO: fake pool
						fake_img = sample_fake_pool()
					with amp.autocast(enabled = enable_fp16) :
						real_logits = network_dis(real_img)
						fake_logits = network_dis(fake_img)
						loss_dis_real = loss_gan(real_logits, 'real')
						loss_dis_fake = loss_gan(fake_logits, 'fake')
						loss_dis = 0.5 * (loss_dis_real + loss_dis_fake)
					if torch.isnan(loss_dis) :
						breakpoint()

					scaler.scale(loss_dis / float(gradient_accumulate)).backward()

					loss_dis_real_meter(loss_dis_real.item())
					loss_dis_fake_meter(loss_dis_fake.item())
					loss_dis_meter(loss_dis.item())
				scaler.unscale_(opt_dis)
				scaler.step(opt_dis)
				scaler.update()
			# train generator
			for gen in range(n_gen) :
				opt_gen.zero_grad()
				for _ in range(gradient_accumulate) :
					real_img, mask = next(dataloader)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					with amp.autocast(enabled = enable_fp16) :
						inpainted_result = network_gen(real_img_masked)
						loss_gen_l1 = F.l1_loss(inpainted_result, real_img)
						loss_gen_fm = 0 # TODO: use VGG19/Darknet53 as perceptual loss
						generator_logits = network_dis(inpainted_result)
						loss_gen_gan = loss_gan(generator_logits, 'generator')
						loss_gen = weight_l1 * loss_gen_l1 + weight_fm * loss_gen_fm + weight_gan * loss_gen_gan
					if torch.isnan(loss_gen) :
						breakpoint()

					scaler.scale(loss_gen / float(gradient_accumulate)).backward()

					loss_gen_meter(loss_gen.item())
					loss_gen_l1_meter(loss_gen_l1.item())
					sch_meter(loss_gen_l1.item()) # use L1 loss as lr scheduler metric
					loss_gen_fm_meter(loss_gen_fm.item())
					loss_gen_gan_meter(loss_gen_gan.item())
				scaler.unscale_(opt_gen)
				scaler.step(opt_gen)
				scaler.update()
			if counter > 0 and counter % record_freq == 0 :
				writer.add_scalar('discriminator/all', loss_dis_meter(reset = True), counter)
				writer.add_scalar('discriminator/real', loss_dis_real_meter(reset = True), counter)
				writer.add_scalar('discriminator/fake', loss_dis_fake_meter(reset = True), counter)
				writer.add_scalar('generator/all', loss_gen_meter(reset = True), counter)
				writer.add_scalar('generator/l1', loss_gen_l1_meter(reset = True), counter)
				writer.add_scalar('generator/fm', loss_gen_fm_meter(reset = True), counter)
				writer.add_scalar('generator/gan', loss_gen_gan_meter(reset = True), counter)
				writer.add_image('original/image', img_unscale(real_img), counter, dataformats = 'NCHW')
				writer.add_image('original/mask', mask, counter, dataformats = 'NCHW')
				writer.add_image('original/masked', img_unscale(real_img_masked), counter, dataformats = 'NCHW')
				writer.add_image('inpainted/image', img_unscale(inpainted_result), counter, dataformats = 'NCHW')
				torch.save(
					{
						'dis': network_dis.state_dict(),
						'gen': network_gen.state_dict(),
						'dis_opt': opt_dis.state_dict(),
						'gen_opt': opt_gen.state_dict(),
					},
					os.path.join(checkpoint_path, 'checkpoints', f'update_{counter}.ckpt')
				)
			if counter > 0 and counter % updates_per_epoch == 0 :
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
			},
			os.path.join(checkpoint_path, 'checkpoints', f'latest.ckpt')
		)
