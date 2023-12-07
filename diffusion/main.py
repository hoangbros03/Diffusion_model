import glob
import sys

import wandb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import imageio

from data import DiffSet
from model import DiffusionModel

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

if __name__=="__main__":
    main()