# dgconv.pytorch
PyTorch implementation of Dynamic Grouping Convolution and Groupable ConvNet in [Differentiable Learning-to-Group Channels via Groupable Convolutional Neural Networks](https://arxiv.org/abs/1908.05867).

<img src="Dynamic_Conv.png" alt="Dynamic_Conv" style="zoom: 25%;" />

* *Kronecker Product* is utilized to construct the sparse matrix efficiently and regularly.
* Discrete optimization is solved with the *Straight-Through Estimator* trick.
* Automatically learn the number of groups in an end-to-end differentiable fashion.

## Usage

**DGConv** is used as a drop-in replacement of depthwise separable convolution in the original ResNeXt to build G-ResNeXt-50/101 network architectures.

```python
from g_resnext import g_resnext50, g_resnext101
model = g_resnext50()
```

## Citation

```bibtex
@InProceedings{Zhang_2019_ICCV,
author = {Zhang, Zhaoyang and Li, Jingyu and Shao, Wenqi and Peng, Zhanglin and Zhang, Ruimao and Wang, Xiaogang and Luo, Ping},
title = {Differentiable Learning-to-Group Channels via Groupable Convolutional Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```

