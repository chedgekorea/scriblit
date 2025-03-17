import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFilter
import numpy as np

class Indoor_dataset(Dataset):
	def __init__(self, tokenizer, dataset):
		self.tokenizer = tokenizer
		self.dataset = dataset
		self.data = []
		with open('./dataset/%s/prompt.json'%self.dataset, 'rt') as f:
			for line in f:
				self.data.append(json.loads(line))

		self.image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
		])

		self.conditioning_image_transforms = transforms.Compose([
			transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.ToTensor(),
		])
		self.transform = transforms.ToTensor()
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = self.data[idx]

		normal_filename = item['normal']
		shading_filename = item['shading']
		target_filename = item['target']
		albedo_filename = item['albedo']
		prompt = item['prompt']
		
		# Read images using PIL
		normal = Image.open(normal_filename).convert('RGB')
		shading = Image.open(shading_filename).convert('RGB')
		target = Image.open(target_filename).convert('RGB')
		albedo = Image.open(albedo_filename).convert('RGB')		

		scribble = shading2scrib(Image.open(shading_filename))
		scribble = scribble.convert('RGB')

		# Apply transforms
		normal = self.conditioning_image_transforms(normal)
		shading = self.conditioning_image_transforms(shading)
		scribble = self.conditioning_image_transforms(scribble)
		target = self.image_transforms(target)
		albedo = self.image_transforms(albedo)
		
		# source = shading
		source = torch.cat((normal, scribble), dim=0)

		control_target = torch.cat((normal, shading), dim=0)

		prompt = self.tokenizer(
			prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
		)
		return dict(pixel_values=target, albedo=albedo, input_ids=prompt.input_ids, conditioning_pixel_values=source, control_target=control_target)



def shading2scrib(image):
	odd_numbers = list(range(3, 20, 2))
	random_odd = random.choice(odd_numbers)

	image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
	image_np = np.array(image)

	#might need to change the parameter a & c
	a = 180
	c = 100

	output_image_np = np.zeros_like(image_np)
	output_image_np[image_np > a] = 255
	output_image_np[image_np < c] = 0
	output_image_np[(image_np <= a) & (image_np >= c)] = 127

	output_image = Image.fromarray(output_image_np)

	dilated_image = output_image.filter(ImageFilter.MinFilter(random_odd))
	eroded_image = dilated_image.filter(ImageFilter.MaxFilter(random_odd))
	return eroded_image