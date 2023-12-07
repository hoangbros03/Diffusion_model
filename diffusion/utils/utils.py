from pathlib import Path

import wandb

def check_and_create_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Create path: {str(path)}")
    else:
        print(f"Path {str(path)} is already existed")

def login_wandb(key):
    wandb.login(key='7801339f18c9b00cf55e8f3c250afa3cba1d141b')

def finish_wandb():
    wandb.finish()