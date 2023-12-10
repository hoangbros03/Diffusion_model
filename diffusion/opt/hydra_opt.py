""" 
Replace the old default_opt.
"""
from omegaconf import OmegaConf
from hydra import compose, initialize


class Opt:
    def __init__(self, flatten=True, verbose=True):
        with initialize(version_base=None, config_path="../conf"):
            self.cfg = compose(config_name="config")
            # self.cfg = OmegaConf.to_container(self.cfg, resolve=True)
        
        if verbose:
            print(f"dct: {self.cfg}")

    def parse_args(self):
        return self.cfg

    