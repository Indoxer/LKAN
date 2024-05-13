# Large Kolmogorov-Arnold Networks
Implementations of KAN variations.

# Installation

### WAY 1 (I don't tested):
installed python 3.10 + nvcc compiler

```
pip install -r requirements.txt
pip install .
```

### The best way:
Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```
conda create -n lkan python==3.10
conda activate lkan
conda install cuda-nvcc
pip install -r requirements.txt
pip install .
```
`pip install .` can take some time first time

# Running

To run mnist select config in `main.py` and run `main.py`.

To view charts, run `tensorboard --logdir ./.experiments/`

# Docs

See examples/

`continual_training_adam.ipynb`, `continual_training_lbfgs.ipynb` - continual training

## Performance (rtx 2060 mobile, mnist):

MLP (31.8M parameters) - 51 it/s 

KANLinear0 (32.3 M parameters) - 4.3 it/s

KANLinear (31M parameters) - 36.5 it/s 

KANLinearFFT (31,1M parameters) - 40 it/s

KANLinearFFT CUDA (30%-50% memory usage compared to KANLinearFFT for forward and backward) = 22 it/s

# Problems
- [ ] update_grid on cuda raise error (torch.linalg.lstsq assume full rank on cuda, only one algorithm) - solved temporary, moved calculating lstsq to cpu
- [ ] update_grid_from_samples in original KAN run model multiple times, is it necessary? 
- [ ] parameters counting, is grid parameter or not?
- [ ] MLP training is almost instant, but KAN train slow on start

# TODO/Ideas:
- [x] Base structure
- [x] KAN simple implementation
- [x] KAN trainer
- [x] train KAN on test dataset
- [ ] remove unnecessary dependencies in requirements.txt
- [ ] test update_grid and "Other possibilities are: (a) the grid is learnable with gradient descent" from paper. 
- [ ] Regularization
- [x] Compare with MLP
- [ ] Grid extension
- [x] MNIST
- [ ] CIFAR10
- [ ] KAN ResNet?
- [x] KAN as CNN filter?
- [ ] KAN in VIT?
- [x] Fourier KAN?
- [ ] GraphKAN
- [ ] Mixing KAN and normal Layers.
- [ ] pruning
- [ ] test continual learning
- [ ] docs and examples - write notebooks like in KAN repo.
- [ ] KAN vs MLP in "LLM" - test?
- [ ] CUDA kernel for b_splines?
- [ ] unit tests?

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


