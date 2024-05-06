# Large Kolmogorov-Arnold Networks
Implementations of KAN variations.

# Installation

```
pip install .
```

# Docs

See examples/ (in future)

# Problems
- [ ] update_grid for MNIST raise errors (big parts of images are zero, so torch.linalg.lstsq raise error)
- [ ] update_grid_from_samples in original KAN run model multiple times, is it necessary? 

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
- [ ] KAN as CNN filter, KAN in VIT?
- [ ] Fourier KAN?
- [ ] pruning
- [ ] test continual learning
- [ ] docs and examples - write notebooks like in KAN repo.
- [ ] KAN vs MLP in "LLM" - test?
- [ ] CUDA kernel for b_splines?

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
Original KAN repo - https://github.com/KindXiaoming/pykan
efficient-kan - https://github.com/Blealtan/efficient-kan
