# svgf-torch

A [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced)(2017) implementation written in PyTorch forward operations(not using backprop or some training features!), for personal study and test purposes. It consists of temporal reprojection/accumulation, variance estimation/filtering and a-trous filtering. This repository is not the official implementation.

- Input: a sequence of path-traced framebuffers(color) and the corresponding g-buffers(world-space normal, world-space position)
- Output: a denoised frame
- Why not shader or numba?: just for fun

## References

- Schied, Christoph, et al. "Spatiotemporal variance-guided filtering: real-time reconstruction for path-traced global illumination." Proceedings of High Performance Graphics. 2017. 1-12.