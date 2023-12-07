""" 
Get gif samples
"""
from pathlib import Path

import torch

from diffusion.model import DiffusionModel
from diffusion.utils import check_and_create_dir

# VARIABLES
diffusion_steps = 1000
dataset_choice="Fashion"
max_epoch=10
batch_size = 128
load_model = False
load_version_num=1

train_dataset_size=32
img_depth = 1

def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)

if __name__=="__main__":
    output_folder = Path("output")
    check_and_create_dir(Path(".") / output_folder)
    model_path= str(Path("e:\New download destination\model_40.pkl"))
    model = DiffusionModel.load_from_checkpoint(model_path, in_size=32*32, t_range=diffusion_steps, img_depth=1)

    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, 1, 32, 32))
    sample_steps = torch.arange(model.t_range-1, 0, -1)
    for t in sample_steps:
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], 32, 32, 1)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)
    