import argparse

class Opt:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Diffusion options")
        self.parser.add_argument("purpose", help="What do you want to do? (train/test/infer)")
        self.parser.add_argument("--diffusion_steps", help="Step of the diffusion to generate the image", type=int, default=1000)
        self.parser.add_argument("--dataset_choice", help="Choice of the dataset", type=str, default='Fashion')
        self.parser.add_argument("--max_epochs", help="Max epoch", type=int, default=10)
        self.parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
        self.parser.add_argument("--device", help="Cuda or cpu?", type=str, default='cpu')
        self.parser.add_argument("--model_path", help="Path contaning the checkpoint", type=str, default=False)
        self.parser.add_argument("--wandb_key", help="Your wandb key", type=str, default=False)

    def parse_args(self):
        return self.parser.parse_args()
