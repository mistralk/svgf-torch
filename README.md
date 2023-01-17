# svgf-torch

A [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced)(2017) implementation written in PyTorch tensor operations(not using backward ops), for personal study and test purposes. It consists of temporal reprojection/accumulation, variance estimation/filtering and a-trous filtering. This repository is not the official implementation.

- Input: a sequence of path-traced framebuffers(color) and the corresponding g-buffers(world-space normal, world-space position)
- Output: a denoised frame
- Why not shader or numba or numpy?: just for fun

When I implemented this, I assumed only for denoising of a specific, limited dataset(static medical images), which is not publicly available. **So some parameters are different with the SVGF paper and some features have not implemented.** Especially, it does not support scenes including dynamic objects with model transformation. (of course, you can easily fix this just by adding matrix multiplication at reprojection step if you need)

## References

- Schied, Christoph, et al. "Spatiotemporal variance-guided filtering: real-time reconstruction for path-traced global illumination." Proceedings of High Performance Graphics. 2017. 1-12.
