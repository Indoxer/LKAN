# Large Kolmogorov-Arnold Networks
Implementations of KAN variations.

# Installation

```
pip install .
```

To run mnist select config in `main.py` and run `main.py`.

To view charts, run `tensorboard --logdir .experiments/`

# Docs

See examples/ (in future)

# Problems
- [ ] update_grid on cuda raise error (torch.linalg.lstsq assume full rank on cuda, only one algorithm) - solved temporary, moved calculating lstsq to cpu
- [ ] update_grid_from_samples in original KAN run model multiple times, is it necessary? 
- [ ] parameters counting, is grid parameter or not?
- [ ] MLP training is almost instant, but KAN train slow on start
- [ ] MLP (around 230 it/s), KAN (around 130 it/s), need optimization. Maybe worse, data flow can be bootleneck for MLP

# TODO/Ideas:
- [x] Base structure
- [x] KAN simple implementation
- [x] KAN trainer
- [x] train KAN on test dataset
- [x] remove unnecessary dependencies in requirements.txt
- [x] test update_grid and "Other possibilities are: (a) the grid is learnable with gradient descent" from paper. 
- [ ] Regularization
- [ ] Compare with MLP
- [ ] Grid extension
- [x] MNIST
- [ ] CIFAR10
- [ ] More datasets?
- [ ] KAN as CNN filter?
- [ ] KAN in VIT?
- [ ] Fourier KAN?
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

[efficient-kan](https://github.com/Blealtan/efficient-kan) - KANLinearB and optimizations


