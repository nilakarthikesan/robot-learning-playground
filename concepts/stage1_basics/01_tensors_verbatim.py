"""The Tensors page, line by line, with nothing added.

Run me:  python concepts/stage1_basics/01_tensors_verbatim.py

Source: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

Every code line below is the tutorial's own, in the tutorial's order, so you can read the
webpage and this file side by side. The only additions are:
  - the STEP banners, so you can find your place in the output
  - a few extra print() calls, each marked "# page shows no output for this line"

Read 01_tensors.py afterwards for the commentary on *why* each of these matters.
"""

import numpy as np
import torch


def step(n, title):
    print(f"\n{'-' * 72}\nSTEP {n}: {title}\n{'-' * 72}")


# ==================================================================================
# Initializing a Tensor
# ==================================================================================

step(1, "Directly from data")

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data: \n {x_data} \n")  # page shows no output for this line


step(2, "From a NumPy array")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np: \n {x_np} \n")  # page shows no output for this line


step(3, "From another tensor")

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


step(4, "With random or constant values")

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# ==================================================================================
# Attributes of a Tensor
# ==================================================================================

step(5, "Attributes of a Tensor")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# ==================================================================================
# Operations on Tensors
# ==================================================================================

step(6, "Moving to the accelerator")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
print(f"Device tensor is stored on: {tensor.device}")  # page shows no output for this line


step(7, "Standard numpy-like indexing and slicing")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)


step(8, "Joining tensors")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


step(9, "Arithmetic operations")

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# page prints only the last expression's value; here are all six so you can compare
print(f"y1 (matmul): \n {y1} \n")
print(f"y2 == y1: {torch.equal(y2, y1)}   y3 == y1: {torch.equal(y3, y1)} \n")
print(f"z1 (elementwise): \n {z1} \n")
print(f"z2 == z1: {torch.equal(z2, z1)}   z3 == z1: {torch.equal(z3, z1)}")


step(10, "Single-element tensors")

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


step(11, "In-place operations")

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


# ==================================================================================
# Bridge with NumPy
# ==================================================================================

step(12, "Tensor to NumPy array")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


step(13, "NumPy array to Tensor")

n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
