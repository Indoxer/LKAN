# Large Kolmogorov-Arnold Networks
Implementations of KAN variations.

# Installation
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install CUDA (If you can run CUDA pytorch code, then it should works)

3. Use `install.sh` (in future `install.cmd`)

# Running

Activate conda env `conda activate lkan`

To run mnist select config in `main.py` and run `main.py`.

To view charts, run `tensorboard --logdir ./.experiments/`

# Docs

See examples/

Only done so far:

- `continual_training_adam.ipynb`, `continual_training_lbfgs.ipynb`

# Contribution

#### Additional development packages/apps:
- cuda-toolkit (nsight compute)

#### Good to know:
1. To run nvidia nsight compute on kernels:
- Add conda python interpreter as executable and python file as args.
- Install lkancpp `CUDA_LINEINFO=1 pip install ./lkancpp/` to see kernels code.


# TODO/Ideas:
- [ ] Use cmake for lkancpp build.
- [ ] remove unnecessary dependencies in requirements.txt
- [ ] test update_grid and "Other possibilities are: (a) the grid is learnable with gradient descent" from paper. 
- [ ] Implement and test (examples notebook) Regularization
- [ ] Implement and test (examples notebook) grid extension
- [ ] MNIST (yaml config to run, model + readme with results)
- [ ] CIFAR10 (yaml config to run, model + readme with results)
- [ ] Test KAN convolution
- [ ] Test KAN as patches encoder in VIT.
- [ ] Implement KAN in latent space (... -> Linear -> KAN -> Linear -> ...)
- [ ] Implement plotting. (plot layers and KAN model)
- [ ] Test scaling behavior on toy dataset (examples notebook)
- [ ] Test scaling behavior on MNIST/CIFAR10 (examples notebook)
- [ ] Test Legendre and Chebyshev polynomials
- [ ] Test Gaussian KAN
- [ ] Write functions to prune KAN layers.
- [ ] Write unit tests for used methods (CUDA version matching pytorch version, pytorch version tests on simple examples)

# Problems
- [ ] update_grid on cuda raise error (torch.linalg.lstsq assume full rank on cuda, only one algorithm) - solved temporary, moved calculating lstsq to cpu
- [ ] update_grid_from_samples in original KAN run model multiple times, is it necessary? 
- [ ] parameters counting, is grid parameter or not?
- [ ] MLP training is almost instant, but KAN train slow on start

# Citations
```python
@misc{liu2024kan,
      title={KAN: Kolmogorov-Arnold Networks}, 
      author={Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark},
      year={2024},
      eprint={2404.19756},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
[Original KAN repo](https://github.com/KindXiaoming/pykan) - base idea

[efficient-kan](https://github.com/Blealtan/efficient-kan) - KANLinear and optimizations

Thanks to [Paluzki]((https://github.com/paluzki)) for proposing Gaussian KAN (KANLinearG - in progress)


