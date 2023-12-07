#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="diffusion",
    version="0.0.1",
    description="Diffusion repo captured what I learned",
    author="Tran Ba Hoang",
    author_email="hoangtb203@gmail.com",
    url="google.com",
    license="MIT",
    install_requires=['lightning', 'hydra-core','wandb'],
    packages=find_packages()
)