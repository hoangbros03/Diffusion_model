import argparse

class Opt:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Diffusion options")
        # TODO: Add other options here

    def parse_args(self):
        return self.parser.parse_args()
