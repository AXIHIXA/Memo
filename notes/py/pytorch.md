# `PyTorch` Notes

- [`PyTorch Tutorials`](https://pytorch.org/tutorials/)

## 🌱 [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### What is PyTorch? 

- It’s a `Python`-based scientific computing package targeted at two sets of audiences: 
    - A replacement for `NumPy` to use the power of GPUs; 
    - a deep learning research platform that provides maximum flexibility and speed. 

### Getting Started

#### Tensors

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
- Get a tensor's size: 
```
>>> x.size()
torch.Size([5, 3])
```
- **NOTE**: `torch.Size` is in fact a `tuple`, so it supports all `tuple` operations. 

#### Operations

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
- Addition: in-place
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
- **NOTE**: Any operation that mutates a tensor in-place is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`. 
- Standard `NumPy`-like indexing with all bells and whistles! 
```
>>> x[:, 1]
tensor([ 1.1034, -0.5884, -0.8236, -0.1217,  0.4772])
```
- Resizing: If you want to resize/reshape tensor, you can use `torch.view`: 
```
>>> x = torch.randn(4, 4)
>>> y = x.view(16)
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> x.size(), y.size(), z.size()
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```
- If you have a one element tensor, use `.item()` to get the value as a `Python` number: 
```
>>> x = torch.randn(1)
>>> x
tensor([0.9551])
>>> x.item()
0.9551321864128113
```
- **READ LATER**: 100+ Tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra, random numbers, etc., are described [here](https://pytorch.org/docs/stable/torch.html). 

### NumPy Bridge

Converting a Torch Tensor to a NumPy array and vice versa is a breeze. 

The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.

#### Converting a Torch Tensor to a NumPy Array

```
>>> a = torch.ones(5)
>>> a
tensor([1., 1., 1., 1., 1.])

>>> b = a.numpy()
>>> b
[1. 1. 1. 1. 1.]
```
See how the numpy array changed in value. 
```
>>> a.add_(1)
>>> a
tensor([2., 2., 2., 2., 2.])
>>> b
[2. 2. 2. 2. 2.]
```

#### Converting NumPy Array to Torch Tensor

All the Tensors on the CPU except a CharTensor support converting to NumPy and back. 
```
>>> a = np.ones(5)
>>> b = torch.from_numpy(a)
>>> np.add(a, 1, out=a)
>>> a
[2. 2. 2. 2. 2.]
>>> b
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

#### CUDA Tensors

Tensors can be moved onto any device using the `.to` method. 
```
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available(): 
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```
Out: 
```
tensor([1.9551], device='cuda:0')
tensor([1.9551], dtype=torch.float64)
```

### Autograd: Automatic Differentiation

Central to all neural networks in PyTorch is the `autograd` package. Let’s first briefly visit this, and we will then go to training our first neural network. 

The `autograd` package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different. 

#### Tensor

`torch.Tensor` is the central class of the package. If you set its attribute `.requires_grad` as `True`, it starts to track all operations on it. When you finish your computation you can call `.backward()` and have all the gradients computed automatically. The gradient for this tensor will be accumulated into `.grad` attribute. 

To stop a tensor from tracking history, you can call `.detach()` to detach it from the computation history, and to prevent future computation from being tracked. 

To prevent tracking history (and using memory), you can also wrap the code block in `with torch.no_grad():`. This can be particularly helpful when evaluating a model because the model may have trainable parameters with `requires_grad=True`, but for which we don’t need the gradients. 

There’s one more class which is very important for autograd implementation: a `Function`. 

`Tensor` and `Function` are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a `.grad_fn` attribute that references a Function that has created the Tensor (**except** for Tensors created by the user; their `grad_fn` is `None`). 

If you want to compute the derivatives, you can call `.backward()` on a `Tensor`. If `Tensor` is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to `backward()`, however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape. 
 
```
>>> x = torch.ones(2, 2, requires_grad=True)
>>> x
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

>>> y = x + 2
>>> y
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
>>> y.grad_fn
<AddBackward0 object at 0x7f67610c4160>

>>> z = y * y * 3
>>> out = z.mean()
>>> z
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
>>> out
tensor(27., grad_fn=<MeanBackward0>)
```

`.requires_grad_(...)` changes an existing Tensor’s `requires_grad` flag in-place. The input flag defaults to `False` if not given. 

```
>>> a = torch.randn(2, 2)
>>> a = ((a * 3) / (a - 1))
>>> a.requires_grad
False

>>> a.requires_grad_(True)
>>> a.requires_grad
True

>>> b = (a * a).sum()
>>> b.grad_fn
<SumBackward0 object at 0x7f67610c4e48>
```

#### Gradients

Let’s backprop now. Because `out` contains a single scalar, `out.backward()` is equivalent to `out.backward(torch.tensor(1.))`. 

```
>>> out.backward()
>>> x.grad
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

Mathematically, if you have a vector valued function `y = f(x)`, then the gradient of `y` with respect to `x` is a Jacobian matrix. Generally speaking, `torch.autograd` is an engine for computing vector-Jacobian product. 

### Neural Networks












































































