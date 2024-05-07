from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="LKAN",
    description="KAN efficient models for easy use in PyTorch.",
    author="Indoxer",
    author_email="indoxer.mk@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
)
