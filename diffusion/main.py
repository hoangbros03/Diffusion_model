import glob
import sys

import wandb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import imageio

from diffusion.data import DiffSet
from diffusion.model import DiffusionModel
from diffusion.opt.default_opt import Opt

# VARIABLES
diffusion_steps = 1000
dataset_choice="Fashion"
max_epoch=10
batch_size = 128
load_model = False
load_version_num=1

pass_version = None
last_checkpoint = None

def main():
    raise NotImplementedError

def test():
    opt = Opt().parse_args()
    print("Parsed Arguments:")
    for arg_name, arg_value in vars(opt).items():
        print(f"{arg_name}: {arg_value}")

if __name__=="__main__":
    test()