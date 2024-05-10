import os

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension

with open("requirements.txt") as f:
    required = f.read().splitlines()

path = os.path.dirname(__file__)
src_folder = "src"
sources = [
    os.path.join(path, src_folder, f)
    for f in os.listdir(os.path.join(path, src_folder))
    if f.endswith(".cpp") or f.endswith(".cu")
]

setup(
    ext_modules=[cpp_extension.CppExtension("kancpp", sources)],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    name="LKAN",
    description="KAN efficient models for easy use in PyTorch.",
    author="Indoxer",
    author_email="indoxer.mk@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=required,
)
