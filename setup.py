import os

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension

with open("requirements.txt") as f:
    required = f.read().splitlines()

required += ["lkancpp"]

setup(
    name="LKAN",
    description="KAN models for easy use in PyTorch.",
    author="Indoxer",
    author_email="indoxer.mk@gmail.com",
    version="0.0.1",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
)
