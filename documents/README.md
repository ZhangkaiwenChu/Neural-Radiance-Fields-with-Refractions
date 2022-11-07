Neural Radiance Fields with Refraction
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project Pitch**

Zhangkaiwen Chu

## Introduction

[Neural Radiance Fields](https://www.matthewtancik.com/nerf) (NeRF) is a functional volumetric repersentation of the 3-D scene. By numeric intergate the radiance field through the ray, the pixel color can be recovered. Since NeRF assumes the light travels a straight line, it cannot due with scenes with reflection and refraction. There are two styles of solutions to this issue. The neural network style approach is to model the blending of the light as an offest with a neural network, and [LB-NERF](https://ieeexplore.ieee.org/document/9897642) is a recent work using this approach. The tradional style approach is to estimate a index of refraction(IoR) field and model the blended light ray with the IoR field, which is tried in this [work](https://dl.acm.org/doi/abs/10.1145/3528233.3530706). 

IB-NERF is not evaluated thoroughly in the paper. They didn't tried the positional-coding and other tricks, and the model is only tested on limited simple scene. Also, the training time is about 11 hours, which is far from realtime. Since they did not released their code or dataset, we are planning to replement their model with pytorch, test various tricks and evaluate the model in more complex scenes, and finally make a CUDA implementation of the model.

However, since there is no gauranteen that their model can perform well in complex scenes, we have a back-up plan. The later [work](https://dl.acm.org/doi/abs/10.1145/3528233.3530706) can produce high quality images, but their algorithm need hand annotation about the space containing the refractive object, and their code is too slow due to the high complexity of their architecture and the overhead of pytorch. Furthermore, they choose Gaussian kernel to estimate the IoR, which is proved to result in a bad estimation of the IoR. So we plan to automate the process of finding the refractive object, experiment with different kernels and transfer at least part of their implementation to CUDA.

## Weekly Timelines

Milestone 1 - Nov. 16
* Implement IB-NERF in pytorch.
* Evaluate the model in real scenes.

#if IB-NERF-GOOD

Milestone 2 - Nov. 28
* Try position encoding.
* Try importance sampling.
* Try other optimizations.

Milestone 3 - Dec. 5
* Implement the optimized model in CUDA.

Final Project Submission - Dec. 11
* Do runtime performance analysis.
* Prepare for the presentation.

#else

Milestone 2 - Nov. 28
* Automate the process of finding the refractive object.
* Try different kenerals.
* Try other optimizations.

Milestone 3 - Dec. 5
* Implement the optimized model in CUDA.

Final Project Submission - Dec. 11
* Do runtime performance analysis.
* Prepare for the presentation.

#endif