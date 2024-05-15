# Performance (rtx 2060 mobile)
###############################################################
    EFFICIENT KAN:
    forward avg time: 9.406 ms
    backward avg time: 10.49 ms
    forward max memory peak: 469.5 MB
    backward max memory peak: 471 MB
    
    batch size: 200
    in dim: 200
    out dim: 200
    grid size: 200
###############################################################
###############################################################
    FFTKAN CUDA:
    forward avg time: 10.09 ms
    backward avg time: 60.13 ms
    forward max memory peak: 148.8 MB
    backward max memory peak: 214.3 MB
    
    batch size: 200
    in dim: 200
    out dim: 200
    grid size: 200
###############################################################
###############################################################
    EFFICIENT KAN:
    forward avg time: 17.41 ms
    backward avg time: 27.25 ms
    forward max memory peak: 2060 MB
    backward max memory peak: 2574 MB
    
    batch size: 1
    in dim: 800
    out dim: 800
    grid size: 100
###############################################################
###############################################################
    FFTKAN CUDA:
    forward avg time: 9.466 ms
    backward avg time: 228.2 ms
    forward max memory peak: 1051 MB
    backward max memory peak: 1568 MB
    
    batch size: 1
    in dim: 800
    out dim: 800
    grid size: 100
###############################################################
###############################################################
    EFFICIENT KAN:
    forward avg time: 4.624 ms
    backward avg time: 2.529 ms
    forward max memory peak: 41.81 MB
    backward max memory peak: 41.59 MB
    
    batch size: 200
    in dim: 200
    out dim: 200
    grid size: 10
###############################################################
###############################################################
    FFTKAN CUDA:
    forward avg time: 1.621 ms
    backward avg time: 3.965 ms
    forward max memory peak: 25.19 MB
    backward max memory peak: 29.62 MB
    
    batch size: 200
    in dim: 200
    out dim: 200
    grid size: 10
###############################################################

## mnist:

MLP (31.8M parameters) - 51 it/s 

KANLinear0 (32.3 M parameters) - 4.3 it/s

KANLinear (31M parameters) - 36.5 it/s 

KANLinearFFT (31,1M parameters) - 40 it/s

KANLinearFFT CUDA (30%-50% memory usage compared to KANLinearFFT for forward and backward) = 22 it/s