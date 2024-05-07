# Large Kolmogorov-Arnold Networks
Implementations of KAN variations.

# Installation

```
pip install .
```

To run mnist select config in `main.py` and run `main.py`.

To view charts, run `tensorboard --logdir .experiments/`

# Info

## Performance (rtx 2060 mobile, mnist):

MLP (31.8M parameters) - 51 it/s 

KANLinear (32.3 M parameters) - 4.3 it/s

KANLinear2 (31M parameters) - 36.5 it/s 

KANLinearFFT (33M parameters) - 40 it/s

## KANConv2d on MNIST:

`test_conv.py` - file with training code

MNIST, batch_size=64, epochs=5, lr=0.0003

Normal CNN (42154 parameters) 98,85% accuracy (130 it/s - memory bootleneck on MNIST, so real performance is better)

Convolution KAN (40050 parameters) - 97,3% accuracy (need hyperparameter tuning or architecture change, KANLinearFFT is problematic in small size because of O(N^2\*L\*2\*G), for small N and L like CNN kernels, 2\*G is significant) (44 it/s - need more optimization)

# Docs

See examples/ (in future)

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
- [ ] Compare with MLP
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

[efficient-kan](https://github.com/Blealtan/efficient-kan) - KANLinear2 and optimizations


