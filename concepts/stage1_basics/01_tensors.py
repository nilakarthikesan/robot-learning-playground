"""Stage 1a: tensors.

Run me:  python concepts/stage1_basics/01_tensors.py

Covers the "Tensors" section of https://docs.pytorch.org/tutorials/beginner/basics/intro.html
The goal is not to memorize the API. It is to internalize four facts:

  1. A tensor is an n-dimensional array that also carries a dtype, a device, and
     (optionally) a record of how it was computed.
  2. Shape is the thing you must always be tracking.
  3. Broadcasting silently changes shapes for you, which is convenient and dangerous.
  4. Some tensors share memory with each other. Views are not copies.
"""

import torch


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


section("1. A tensor has a shape, a dtype, and a device")

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"x =\n{x}")
print(f"shape  {tuple(x.shape)}   dtype {x.dtype}   device {x.device}")
print(
    "\nShape (2, 3) means 2 rows of 3. Read it right-to-left as 'innermost first':\n"
    "the last dimension is the one whose elements sit next to each other in memory."
)

print(f"\nints are a different dtype: {torch.tensor([1, 2, 3]).dtype}")
print(
    "This matters. Integer tensors cannot be differentiated, and a stray int tensor is a\n"
    "very common cause of 'expected scalar type Float but found Long' errors."
)

if torch.backends.mps.is_available():
    print("\nApple GPU (MPS) is available. x.to('mps') would move this tensor to it.")
else:
    print("\nNo GPU backend found; everything here runs on CPU, which is plenty.")


section("2. Shapes: the operations you will use constantly")

t = torch.arange(12).reshape(3, 4)
print(f"torch.arange(12).reshape(3, 4) =\n{t}")
print(f"\nt.T (transpose)          -> {tuple(t.T.shape)}")
print(f"t.reshape(2, 6)          -> {tuple(t.reshape(2, 6).shape)}")
print(f"t.reshape(-1)            -> {tuple(t.reshape(-1).shape)}   (-1 means 'infer this')")
print(f"t.unsqueeze(0)           -> {tuple(t.unsqueeze(0).shape)}   (insert a size-1 axis)")
print(f"t.unsqueeze(0).squeeze() -> {tuple(t.unsqueeze(0).squeeze().shape)}  (drop size-1 axes)")

print(
    "\nunsqueeze/squeeze look like busywork now. In stage 3 you will see them on almost\n"
    "every line, because attention requires lining up a (batch, 1, hidden) query against\n"
    "a (batch, seq_len, hidden) set of keys, and the size-1 axis is what makes that work."
)


section("3. Matrix multiply vs elementwise multiply")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(f"a * b   (elementwise) =\n{a * b}")
print(f"\na @ b   (matmul) =\n{a @ b}")
print(
    "\nConfusing these two is the classic beginner bug, and it will not always raise an\n"
    "error -- for square matrices both are legal and you just get wrong numbers."
)

batch_a = torch.randn(8, 3, 4)
batch_b = torch.randn(8, 4, 5)
print(f"\nbatched: (8, 3, 4) @ (8, 4, 5) -> {tuple((batch_a @ batch_b).shape)}")
print(
    "matmul treats all leading dimensions as batch and multiplies the last two.\n"
    "torch.bmm does the same thing for exactly 3D inputs; you will meet it in attention."
)


section("4. Broadcasting")

m = torch.zeros(3, 4)
row = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"(3, 4) + (4,) -> {tuple((m + row).shape)}")
print(f"{m + row}")
print(
    "\nThe (4,) vector was stretched across all 3 rows without being copied.\n"
    "Rule: align shapes from the right; each pair of dims must be equal, or one must be 1."
)

col = torch.tensor([[1.0], [2.0], [3.0]])
print(f"\n(3, 1) + (1, 4) -> {tuple((col + row.reshape(1, 4)).shape)}  <- both got stretched")
print(
    "\nThe danger: a (3, 1) plus a (3,) broadcasts to (3, 3) instead of erroring. If your\n"
    "loss is suspiciously shaped or your model 'trains' but never improves, check for this."
)


section("5. Views share memory; clones do not")

original = torch.zeros(2, 3)
view = original.reshape(3, 2)
view[0, 0] = 99.0
print(f"after writing to the reshaped tensor, the original is:\n{original}")
print("reshape gave a *view*: the same storage, read with different strides.")

copy = original.clone()
copy[0, 1] = -1.0
print(f"\nafter writing to a clone, the original is unchanged:\n{original}")


section("6. Reductions and the dim argument")

s = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(f"s =\n{s}")
print(f"\ns.sum()        = {s.sum()}                (everything -> a scalar)")
print(f"s.sum(dim=0)   = {s.sum(dim=0)}   (collapse rows    -> shape {tuple(s.sum(dim=0).shape)})")
print(f"s.sum(dim=1)   = {s.sum(dim=1)}       (collapse cols -> shape {tuple(s.sum(dim=1).shape)})")
print(
    "\nMental model for `dim=k`: the dimension you name is the one that disappears.\n"
    "(Unless you pass keepdim=True, which leaves it behind with size 1.)"
)

print(f"\ns.softmax(dim=1) =\n{s.softmax(dim=1)}")
print("Each row now sums to 1. Softmax over the wrong dim is a silent, nasty bug.")


section("What to take away")
print(
    """
Almost every error you hit in the next two stages will be a shape error. The habit that
prevents them is boring and effective: print the shape of anything you are unsure about,
immediately, before moving on.

Next: 02_autograd.py, where tensors start remembering where they came from.
"""
)
