import glob
import sys

import wandb
import torch
import imageio

from diffusion.opt.default_opt import Opt
from diffusion.utils.utils import login_wandb, finish_wandb
from diffusion.train import train

def main():
    opt = Opt().parse_args()
    if bool(opt.wandb_key):
        login_wandb(opt.wandb_key)
    if opt.purpose == "train":
        train(opt)
    else:
        raise ValueError("Not support your purpose yet!")
    if bool(opt.wandb_key):
        finish_wandb

def test():
    opt = Opt().parse_args()
    print("Parsed Arguments:")
    for arg_name, arg_value in vars(opt).items():
        print(f"{arg_name}: {arg_value}")

if __name__=="__main__":
    main()