import os
import sys

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension

path = os.path.dirname(__file__)
src_folder = "."
sources = []

for root, dirs, files in os.walk(os.path.join(path, src_folder)):
    for file in files:
        if file.endswith(".cu") or file.endswith(".cpp"):
            sources.append(os.path.join(root, file))

extra_compile_args = {"cxx": ["-g"]}

if os.environ.get("CUDA_LINEINFO", False):
    extra_compile_args["nvcc"] = ["-lineinfo"]

setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            "lkancpp", sources, extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    name="lkancpp",
    description="KAN for CUDA and CPU",
    author="Indoxer",
    author_email="indoxer.mk@gmail.com",
    version="0.0.1",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
)
