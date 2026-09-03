"""Stage 1a: tensors.

Run me:  python concepts/stage1_basics/01_tensors.py

Covers https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
Parts 1-4 follow the tutorial's four sections in order. Parts 5-8 are the things the
tutorial mentions in passing that turn out to cause most real bugs.

A tensor is an n-dimensional array that also carries a dtype, a device, and (optionally)
a record of how it was computed. Four facts are worth internalizing:

  1. Shape is the thing you must always be tracking.
  2. dtype and device are silent failure modes: mismatches error, or worse, don't.
  3. Broadcasting changes shapes for you, which is convenient and dangerous.
  4. Some tensors share memory with each other. Views are not copies.
"""

import numpy as np
import torch

# Fixed so that two runs print the same "random" numbers and you can compare them.
torch.manual_seed(0)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------------------
section("1. Initializing a tensor")

# Directly from Python data. The dtype is inferred.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"from a nested list:\n{x_data}   dtype={x_data.dtype}")
print(
    "Note it inferred int64, not float. Integer tensors cannot be differentiated, and a\n"
    "stray int tensor is a very common cause of 'expected scalar type Float but found Long'."
)

# From a NumPy array.
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"\nfrom a numpy array:\n{x_np}   dtype={x_np.dtype}")

# From another tensor: shape and dtype are inherited unless overridden.
x_ones = torch.ones_like(x_data)
print(f"\ntorch.ones_like(x_data):\n{x_ones}   dtype={x_ones.dtype}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"\ntorch.rand_like(x_data, dtype=torch.float):\n{x_rand}")
print(
    "The dtype override is not optional here: rand_like on an int64 tensor would try to\n"
    "put uniform [0, 1) samples into an integer container and raise."
)

# From an explicit shape.
shape = (2, 3)
print(f"\ntorch.rand({shape}):\n{torch.rand(shape)}")
print(f"\ntorch.ones({shape}):\n{torch.ones(shape)}")
print(f"\ntorch.zeros({shape}):\n{torch.zeros(shape)}")


# ----------------------------------------------------------------------------------
section("2. Attributes of a tensor")

tensor = torch.rand(3, 4)
print(f"shape    {tuple(tensor.shape)}")
print(f"dtype    {tensor.dtype}")
print(f"device   {tensor.device}")
print(
    "\nThose three attributes are the whole identity of a tensor for debugging purposes.\n"
    "Shape (3, 4) means 3 rows of 4. Read shapes right-to-left as 'innermost first': the\n"
    "last dimension is the one whose elements sit next to each other in memory."
)


# ----------------------------------------------------------------------------------
section("3. Operations on tensors")

print("--- moving to an accelerator ---")
if torch.accelerator.is_available():
    acc = torch.accelerator.current_accelerator()
    on_gpu = tensor.to(acc)
    print(f"accelerator found: {acc}")
    print(f"tensor.to({acc}).device -> {on_gpu.device}")
    print(
        "\nOn this Mac that is MPS, Apple's Metal backend. Tensors are created on the CPU by\n"
        "default and must be moved explicitly; copying large ones across devices is not free.\n"
        "Everything below stays on the CPU deliberately, so device never becomes a variable."
    )
else:
    print("No accelerator; everything runs on CPU, which is plenty for this repo.")

print("\n--- indexing and slicing (same rules as numpy) ---")
tensor = torch.ones(4, 4)
print(f"First row:    {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column:  {tensor[..., -1]}")
tensor[:, 1] = 0
print(f"after tensor[:, 1] = 0:\n{tensor}")
print(
    "`...` means 'all the remaining leading dimensions'. It is how you write code that\n"
    "works whether or not a batch dimension is present."
)

print("\n--- joining ---")
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"torch.cat([t, t, t], dim=1) -> {tuple(t1.shape)}")
t2 = torch.stack([tensor, tensor, tensor], dim=0)
print(f"torch.stack([t, t, t], dim=0) -> {tuple(t2.shape)}")
print(
    "\nThe difference is the one people get wrong: cat glues along an axis that already\n"
    "exists (4x4 -> 4x12), stack creates a new axis (4x4 -> 3x4x4). 'Collate a batch' is\n"
    "stack; 'append more features' is cat."
)

print("\n--- arithmetic: matmul vs elementwise ---")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"tensor @ tensor.T =\n{y1}")
print(f"all three matmul spellings agree: {torch.equal(y1, y2) and torch.equal(y1, y3)}")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"\ntensor * tensor (elementwise) =\n{z1}")
print(f"all three mul spellings agree: {torch.equal(z1, z2) and torch.equal(z1, z3)}")
print(
    "\nConfusing @ with * is the classic beginner bug, and it does not always raise -- for\n"
    "square matrices both are legal and you simply get wrong numbers."
)

batch_a = torch.randn(8, 3, 4)
batch_b = torch.randn(8, 4, 5)
print(f"\nbatched: (8, 3, 4) @ (8, 4, 5) -> {tuple((batch_a @ batch_b).shape)}")
print(
    "matmul treats all leading dimensions as batch and multiplies the last two.\n"
    "torch.bmm does the same for exactly-3D inputs; you will meet it in attention."
)

print("\n--- single-element tensors ---")
agg = tensor.sum()
agg_item = agg.item()
print(f"tensor.sum() = {agg}  (still a tensor)")
print(f".item()      = {agg_item}  ({type(agg_item).__name__})")
print(
    "Use .item() when you want to log or compare a value in Python. Accumulating the\n"
    "tensor instead of the float is how you accidentally keep an entire training run's\n"
    "computation graph alive and run out of memory."
)

print("\n--- in-place operations ---")
print(f"before:\n{tensor}")
tensor.add_(5)
print(f"after tensor.add_(5):\n{tensor}")
print(
    "The trailing underscore means 'mutate the receiver'. It saves memory but destroys the\n"
    "history autograd needs, so PyTorch discourages it inside models. Recognize the\n"
    "convention (add_, copy_, zero_, t_) so you know when a function is mutating its input."
)


# ----------------------------------------------------------------------------------
section("4. Bridge with NumPy")

t = torch.ones(5)
n = t.numpy()
print(f"t: {t}\nn: {n}")

t.add_(1)
print(f"\nafter t.add_(1), the numpy array changed too:\nt: {t}\nn: {n}")

n2 = np.ones(5)
t2 = torch.from_numpy(n2)
np.add(n2, 1, out=n2)
print(f"\nand in the other direction, after np.add(n2, 1, out=n2):\nt2: {t2}\nn2: {n2}")

print(
    "\nA CPU tensor and its numpy array share one buffer; there is no copy and no\n"
    "synchronization. That is a feature (zero-cost interop with scipy, matplotlib,\n"
    "opencv) and a trap (a 'copy' you never made, mutating under you).\n"
    "Note this only works on the CPU. On a GPU tensor .numpy() raises, and you have to\n"
    "write .cpu().numpy() -- which really does copy."
)


# ----------------------------------------------------------------------------------
section("5. Beyond the tutorial: shape surgery")

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


# ----------------------------------------------------------------------------------
section("6. Beyond the tutorial: broadcasting")

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


# ----------------------------------------------------------------------------------
section("7. Beyond the tutorial: views share memory, clones do not")

original = torch.zeros(2, 3)
view = original.reshape(3, 2)
view[0, 0] = 99.0
print(f"after writing to the reshaped tensor, the original is:\n{original}")
print("reshape gave a *view*: the same storage, read with different strides.")

copy = original.clone()
copy[0, 1] = -1.0
print(f"\nafter writing to a clone, the original is unchanged:\n{original}")


# ----------------------------------------------------------------------------------
section("8. Beyond the tutorial: reductions and the dim argument")

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


# ----------------------------------------------------------------------------------
section("What to take away")
print(
    """
Almost every error you hit in the next two stages will be a shape error. The habit that
prevents them is boring and effective: print the shape of anything you are unsure about,
immediately, before moving on.

Next: 02_datasets_dataloaders.py, where tensors start arriving in batches.
"""
)
