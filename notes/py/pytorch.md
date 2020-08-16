# `PyTorch` Notes

- [`PyTorch Tutorials`](https://pytorch.org/tutorials/)

## 🌱 [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### What is PyTorch? 

- It’s a Python-based scientific computing package targeted at two sets of audiences: 
    - A replacement for `NumPy` to use the power of GPUs; 
    - a deep learning research platform that provides maximum flexibility and speed. 

### Tensors

- Construct a `5x3` matrix, uninitialized: 
```
>>> import torch

>>> torch.empty(5, 3)
tensor([[ 5.0375e+28,  4.5625e-41,  5.0195e+28],
        [ 4.5625e-41, -1.8338e+30,  3.0840e-41],
        [-9.9826e+08,  4.5625e-41, -1.5343e+08],
        [ 4.5625e-41, -1.0043e+09,  4.5625e-41],
        [-2.0086e+08,  4.5625e-41, -9.5843e+08]])
```
- Construct a randomly initialized matrix: 
```
>>> torch.rand(5, 3)
tensor([[0.5814, 0.0997, 0.1744],
        [0.2834, 0.9581, 0.9954],
        [0.9372, 0.4401, 0.1696],
        [0.1424, 0.5370, 0.9970],
        [0.6686, 0.5558, 0.5354]])
```
- Construct a matrix filled zeros and of dtype `long`: 
```
>>> torch.zeros(5, 3, dtype=torch.long)
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```
- Construct a tensor directly from data: 
```
>>> torch.tensor([5.5, 3])
tensor([5.5000, 3.0000])
```
- Create a tensor based on an existing tensor: 
```
>>> x = torch.zeros(5, 3, dtype=torch.doouble)
>>> x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
>>> x
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)

>>> torch.randn_like(x, dtype=torch.float)        # override dtype, result has the same size
tensor([[-0.0970,  1.1034, -1.6941],
        [-1.7651, -0.5884, -1.1931],
        [ 1.0376, -0.8236,  0.8907],
        [ 0.4683, -0.1217,  1.2467],
        [ 0.4624,  0.4772, -1.0577]])
```
- Get a tensor's size (`torch.Size` is in fact a `tuple`, so it supports all `tuple` operations): 
```
>>> x.size()
torch.Size([5, 3])
```

### Operations

- There are multiple syntaxes for operations. In the following example, we will take a look at the addition operation. 
- Addition: syntax 1
```
>>> y = torch.rand(5, 3)
>>> x + y
tensor([[ 0.4675,  1.7053, -1.2332],
        [-1.6031,  0.1028, -0.6090],
        [ 1.8545, -0.6408,  1.0778],
        [ 0.9508,  0.7635,  2.1345],
        [ 0.5693,  1.0205, -0.5476]])
```
- Addition: syntax 2
```
>>> torch.add(x, y)
tensor([[ 0.4675,  1.7053, -1.2332],
        [-1.6031,  0.1028, -0.6090],
        [ 1.8545, -0.6408,  1.0778],
        [ 0.9508,  0.7635,  2.1345],
        [ 0.5693,  1.0205, -0.5476]])
```
- Addition: providing an output tensor as argument
```
>>> result = torch.empty(5, 3)
>>> torch.add(x, y, out=result)
>>> result
tensor([[ 0.4675,  1.7053, -1.2332],
        [-1.6031,  0.1028, -0.6090],
        [ 1.8545, -0.6408,  1.0778],
        [ 0.9508,  0.7635,  2.1345],
        [ 0.5693,  1.0205, -0.5476]])
```
- Addition: in-place (Any operation that mutates a tensor in-place is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`.)
```
>>> # adds x to y
>>> y.add_(x)
>>> y
tensor([[ 0.4675,  1.7053, -1.2332],
        [-1.6031,  0.1028, -0.6090],
        [ 1.8545, -0.6408,  1.0778],
        [ 0.9508,  0.7635,  2.1345],
        [ 0.5693,  1.0205, -0.5476]])
```







































































