Neural Raidance Fields with Refractions
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Zhangkaiwen Chu
  * [LinkedIn](https://www.linkedin.com/in/zhangkaiwen-chu-b53060225/)
* Tested on: Windows 10, R7-5800H @ 3.20GHz 16GB, RTX 3070 Laptop GPU 16310MB (Personal Laptop)

This project implement the light-bend neural radiance field (LB-NeRF) which is based on the paper ["LB-NERF: Light Bending Neural Radiance Fields for Transparent Medium"](https://ieeexplore.ieee.org/document/9897642). It aims at modeling the refraction effect with an offset field using a neural network.

## Model Architecture
![](documents/pic/network%20architecture.png)
Here G is the offset network, the input is the position and the view direction, the output is the offset. F is the NeRF network, the input is the position added with the offset and the view direction, the output is the color map and the density.

To better capture the details of the scene, positional endocing is also adopted to the position and the view angle.

## Usage
The pytorch implemententation locates in the code direcctory, ended with .ipynb.
For c++ implementation, we are using the architecture of [instant-npg](https://github.com/NVlabs/instant-ngp), please refer to this page for details. 

## Results
![](documents/video/Ball_spiral_200000_rgb.mp4)