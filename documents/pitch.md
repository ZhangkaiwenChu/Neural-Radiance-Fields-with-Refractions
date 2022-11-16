Neural Radiance Fields with Refraction
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project Pitch**

Zhangkaiwen Chu

## Introduction

[Neural Radiance Fields](https://www.matthewtancik.com/nerf) (NeRF) is a functional volumetric repersentation of the 3-D scene, which can be learned with several photos of the scene. By numeric intergate the radiance field through the ray, the pixel color can be recovered. Since NeRF assumes the light travels a straight line, it cannot due with scenes with reflection and refraction. A neural network style solution to this issue is to model the blending of the light as an offest with a neural network, and [LB-NERF](https://ieeexplore.ieee.org/document/9897642) is a recent work using this approach. 

However, LB-NERF is not evaluated thoroughly in the paper. They didn't tried the positional-coding and other tricks, and the model is only tested on limited simple scene. Also, the training time is about 11 hours, which is far from realtime. Since they did not released their code or dataset, we are planning to reimplement their model with pytorch, test various tricks and evaluate the model in more complex scenes, and finally make a CUDA implementation of the model.



## Weekly Timelines

Milestone 1 - Nov. 16
* Implement LB-NERF in pytorch.
* Evaluate the model in real scenes.

Milestone 2 - Nov. 28
* Try position encoding.
* Try importance sampling.
* Try other optimizations.

Milestone 3 - Dec. 5
* Implement the optimized model in CUDA.

Final Project Submission - Dec. 11
* Do runtime performance analysis.
* Prepare for the presentation.


## Reference

[Neural Radiance Fields](https://www.matthewtancik.com/nerf)

[LB-NERF](https://ieeexplore.ieee.org/document/9897642)

[Eikonal Fields for Refractive Novel-View Synthesis](https://dl.acm.org/doi/abs/10.1145/3528233.3530706)