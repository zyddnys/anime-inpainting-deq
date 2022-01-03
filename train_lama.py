
import os
import time
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
import ffc as models
import vgg_loss

from tqdm import tqdm

def mask_image(img, mask) :
	return img * (1. - mask)

def img_unscale(img) :
	return (torch.clip(img.detach(), -1, 1) + 1) * .5

def train(
	network_gen: nn.Module,
	network_dis: nn.Module,
	dataloader,
	checkpoint_path,
	weight_gan = 0.01,
	weight_l1 = 1.0,
	weight_fm = 10.0,
	weight_vgg = 10.0,
	device = torch.device('cuda:0'),
	n_critic = 3,
	n_gen = 1,
	fake_pool_size = 256,
	lr_gen = 1e-4,
	lr_dis = 5e-4,
	updates_per_epoch = 10000,
	record_freq = 1000,
	total_updates = 1000000,
	gradient_accumulate = 4,
	enable_fp16 = False,
	resume = False
	) :
	if enable_fp16 :
		print(' -- FP16 AMP enabled')
	print(' -- Initializing losses')
	loss_gan = ganloss.GANLossSoftLS(device)
	loss_vgg = vgg_loss.VGG19LossWithStyle().to(device)

	opt_gen = optim.Adam(network_gen.parameters(), lr = lr_gen, betas = (0.5, 0.99), weight_decay = 1e-6)
	opt_dis = optim.Adam(network_dis.parameters(), lr = lr_dis, betas = (0.5, 0.99), weight_decay = 1e-6)

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
	loss_gen_vgg_meter = utils.AvgMeter()
	loss_gen_vgg_coarse_meter = utils.AvgMeter()
	loss_gen_gan_meter = utils.AvgMeter()
	loss_gen_fm_meter = utils.AvgMeter()
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
			dataloader_meter = utils.AvgMeter()
			# train discrimiantor
			for critic in range(n_critic) :
				opt_dis.zero_grad()
				for _ in range(gradient_accumulate) :
					start_time = time.time()
					real_img, mask = next(dataloader)
					end_time = time.time()
					dataloader_meter(end_time - start_time)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					if np.random.randint(0, 2) == 0 or not fakepool.available() :
						with torch.no_grad(), amp.autocast(enabled = enable_fp16) :
							fake_img = network_gen(real_img_masked, mask)
						fakepool.put(fake_img)
					else :
						fake_img = fakepool.sample()
					with amp.autocast(enabled = enable_fp16) :
						real_logits, _ = network_dis(real_img)
						fake_logits, _ = network_dis(fake_img)
						loss_dis_real = loss_gan(real_logits, 'real', None)
						mask_inv = 1 - F.interpolate(mask, size = (real_logits.shape[2], real_logits.shape[3]), mode = 'bicubic', align_corners = False)
						loss_dis_fake = loss_gan(fake_logits, 'fake', mask_inv)
						loss_dis = 0.5 * (loss_dis_real + loss_dis_fake)
					if torch.isnan(loss_dis) or torch.isinf(loss_dis) :
						raise Exception

					scaler_dis.scale(loss_dis / float(gradient_accumulate)).backward()

					loss_dis_real_meter(loss_dis_real.item())
					loss_dis_fake_meter(loss_dis_fake.item())
					loss_dis_meter(loss_dis.item())
				scaler_dis.unscale_(opt_dis)
				#torch.nn.utils.clip_grad_norm_(network_dis.parameters(), 0.1)
				scaler_dis.step(opt_dis)
				scaler_dis.update()
			# train generator
			for gen in range(n_gen) :
				opt_gen.zero_grad()
				for _ in range(gradient_accumulate) :
					start_time = time.time()
					real_img, mask = next(dataloader)
					end_time = time.time()
					dataloader_meter(end_time - start_time)
					real_img, mask = real_img.to(device), mask.to(device)
					real_img_masked = mask_image(real_img, mask)
					with amp.autocast(enabled = enable_fp16) :
						inpainted_result = network_gen(real_img_masked, mask)
						#inpainted_result_coarse, inpainted_result = network_gen(real_img_masked, mask)
						loss_gen_l1 = F.l1_loss(inpainted_result, real_img)
						loss_vgg_combined = loss_vgg(inpainted_result, real_img)
						generator_logits, dis_features_fake = network_dis(inpainted_result)
						with torch.no_grad() :
							_, dis_features_real = network_dis(real_img)
						loss_fm = 0
						for (fm_fake, fm_real) in zip(dis_features_fake, dis_features_real) :
							loss_fm += F.mse_loss(fm_fake, fm_real)
						loss_fm = loss_fm / len(dis_features_real)
						loss_gen_gan = loss_gan(generator_logits, 'generator', None)
						loss_gen = weight_l1 * (loss_gen_l1) + weight_fm * loss_fm + weight_vgg * (loss_vgg_combined) + weight_gan * loss_gen_gan
					if torch.isnan(loss_gen) or torch.isinf(loss_dis) :
						raise Exception

					scaler_gen.scale(loss_gen / float(gradient_accumulate)).backward()

					loss_gen_meter(loss_gen.item())
					loss_gen_l1_meter(loss_gen_l1.item())
					sch_meter(loss_gen_l1.item()) # use L1 loss as lr scheduler metric
					loss_gen_vgg_meter(loss_vgg_combined.item())
					loss_gen_gan_meter(loss_gen_gan.item())
					loss_gen_fm_meter(loss_fm.item())
				scaler_gen.unscale_(opt_gen)
				#torch.nn.utils.clip_grad_norm_(network_gen.parameters(), 0.1)
				scaler_gen.step(opt_gen)
				scaler_gen.update()
			if counter % record_freq == 0 :
				tqdm.write(f' -- Record at update {counter}')
				writer.add_scalar('discriminator/all', loss_dis_meter(reset = True), counter)
				writer.add_scalar('discriminator/real', loss_dis_real_meter(reset = True), counter)
				writer.add_scalar('discriminator/fake', loss_dis_fake_meter(reset = True), counter)
				writer.add_scalar('generator/all', loss_gen_meter(reset = True), counter)
				writer.add_scalar('generator/l1', loss_gen_l1_meter(reset = True), counter)
				writer.add_scalar('generator/vgg', loss_gen_vgg_meter(reset = True), counter)
				writer.add_scalar('generator/gan', loss_gen_gan_meter(reset = True), counter)
				writer.add_scalar('generator/fm', loss_gen_fm_meter(reset = True), counter)
				writer.add_image('original/image', img_unscale(real_img), counter, dataformats = 'NCHW')
				writer.add_image('original/mask', mask, counter, dataformats = 'NCHW')
				writer.add_image('original/masked', img_unscale(real_img_masked), counter, dataformats = 'NCHW')
				writer.add_image('inpainted/refined', img_unscale(inpainted_result), counter, dataformats = 'NCHW')
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
			#tqdm.write(f'Dataloader overhead avg {int(dataloader_meter(reset = True) * 1000)}ms')
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

def main(args, device, enable_fp16 = True) :
	print(' -- Initializing models')
	gen = models.get_generator().to(device)
	dis = models.get_discriminator().to(device)
	ds = dataset.FileListDataset('train.flist', image_size_min = args.image_file_size_min, image_size_max = args.image_file_size_max, patch_size = args.image_size)
	loader = torch.utils.data.DataLoader(
		ds,
		batch_size = args.batch_size,
		num_workers = args.workers,
		worker_init_fn = dataset.init_worker,
		pin_memory = True
	)
	train(gen, dis, loader, args.checkpoint_dir,
		gradient_accumulate = args.gradient_accumulate,
		resume = args.resume,
		enable_fp16 = enable_fp16,
		n_critic = args.num_critic,
		n_gen = args.num_gen,
	)

if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint-dir', '-d', type = str, default = './checkpoints_aot', help = "where to place checkpoints")
	parser.add_argument('--batch-size', type = int, default = 4, help = "training batch size")
	parser.add_argument('--resume', action = 'store_true', help = "resume training")
	parser.add_argument('--disable-amp', action = 'store_true', help = "disable amp fp16 training")
	parser.add_argument('--enable-tf32', action = 'store_true', help = "enable tf32 training for NVIDIA Ampere GPU")
	parser.add_argument('--gradient-accumulate', type = int, default = 8, help = "gradient accumulate")
	parser.add_argument('--image-size', type = int, default = 320, help = "size of cropped patch used for training")
	parser.add_argument('--image-file-size-min', type = int, default = 640, help = "lower bound of smallest axis of image before cropping")
	parser.add_argument('--image-file-size-max', type = int, default = 1280, help = "upper bound of smallest axis of image before cropping")
	parser.add_argument('--workers', type = int, default = 24, help = "num of dataloader workers")
	parser.add_argument('--num-critic', type = int, default = 1, help = "num of critic updates per update")
	parser.add_argument('--num-gen', type = int, default = 1, help = "num of generator updates per update")
	args = parser.parse_args()
	enable_fp16 = not args.disable_amp
	if args.enable_tf32 :
		print(' -- TF32 enabled')
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	main(args, torch.device("cuda:0"), enable_fp16)
