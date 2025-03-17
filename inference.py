from diffusers.utils import load_image
import torch
from PIL import Image
import torchvision.transforms as transforms
from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	StableDiffusionControlNetPipeline,
	UNet2DConditionModel,
	UniPCMultistepScheduler,
	DDIMScheduler
)
import os
import json
from network_controlnet import ControlNetModel
from pipeline_cn import CustomControlNetPipeline
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser(description='Training Monitor Lighting on Various Dataset.')
parser.add_argument('-n', '--n', type=str, help=(
			"pretrained model path"
		))
parser.add_argument('-data', '--data', type=str, default='openroom')
parser.add_argument('-seed', '--seed', type=int, default=6071)
args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=0 python inference.py -n scribblelight_controlnet -data data

def image_to_tensor(image):
	image = image / 255.0
	image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
	return image.float()

def replace_unet_conv_in(unet):
	# replace the first layer to accept 8 in_channels
	_weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
	_bias = unet.conv_in.bias.clone()  # [320]
	_weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)

	# new conv_in channel
	_n_convin_out_channel = unet.conv_in.out_channels
	_new_conv_in = Conv2d(
		8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
	)
	_new_conv_in.weight = Parameter(_weight)
	_new_conv_in.bias = Parameter(_bias)
	unet.conv_in = _new_conv_in
	
	# replace config
	unet.config["in_channels"] = 8
	return

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return torch.manual_seed(seed)

base_model_path = "stabilityai/stable-diffusion-2-1"
experiment = args.n
best_weight = os.listdir(experiment)
best_weight = [f for f in best_weight if f.startswith('checkpoint')]
best_weight = natsorted(best_weight)[-1]
print(best_weight)
controlnet_path = "%s/%s/controlnet"%(experiment, best_weight)
unet_path = "%s/%s/custom_unet.pth"%(experiment, best_weight)

unet = UNet2DConditionModel.from_pretrained(
	base_model_path, subfolder="unet", revision=None, variant=None
)

replace_unet_conv_in(unet)
unet.load_state_dict(torch.load(unet_path), strict=False)
unet.cuda().eval()

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
pipe = CustomControlNetPipeline.from_pretrained(
	base_model_path, controlnet=controlnet, torch_dtype=torch.float32, unet = unet
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

image_transforms = transforms.Compose([
	transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

conditioning_image_transforms = transforms.Compose([
	transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
	transforms.ToTensor(),
])

vae = AutoencoderKL.from_pretrained(
	base_model_path, subfolder="vae", revision=None, variant=None
)
noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

file_path = 'dataset/%s/prompt.json'%args.data


num = 0

if not os.path.exists('./inference/%s'%args.data):
	os.makedirs('./inference/%s'%args.data)

weight_dtype = torch.float32
with open(file_path, 'r') as file:
	for number, line in enumerate(file):
		data = json.loads(line)
		
		normal = data.get('normal')
		shading = data.get('shading')
		albedo = data.get('albedo')
		target = data.get('target')
		prompt = data.get('prompt')
		control_image_normal = load_image(normal)
		control_image_shading = load_image(shading)
		control_image_albedo = load_image(albedo)

		control_image_normal = control_image_normal.resize((512,512))
		control_image_shading = control_image_shading.resize((512,512))

		image = image_transforms(control_image_albedo)
		image = image.unsqueeze(0)
		albedo_latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()
		albedo_latents = albedo_latents * vae.config.scaling_factor
		albedo_noise = torch.randn_like(albedo_latents)#pls
		timesteps_albedo = torch.randint(200, 201, (1,))
		albedo_latents = noise_scheduler.add_noise(albedo_latents, albedo_noise, timesteps_albedo)

		# generate image
		generator = set_seed(args.seed)
		image = pipe(
			prompt, num_inference_steps=20, generator=generator, image=(control_image_normal, control_image_shading), albedo_latents= albedo_latents.cuda()
		).images[0]

		image.save("./inference/%s/output_%d_%d.png"%(args.data,number,args.seed))

		normal = Image.open(normal)
		normal.save("./inference/%s/normal_%d.png"%(args.data,number))
		shading = Image.open(shading).convert('RGB')
		shading.save("./inference/%s/shading_%d.png"%(args.data,number))
		albedo = Image.open(albedo)
		albedo.save("./inference/%s/albedo_%d.png"%(args.data,number))
		target = Image.open(target)
		target.save("./inference/%s/target_%d.png"%(args.data,number))
