"""Stage 1b: autograd.

Run me:  python concepts/stage1_basics/02_autograd.py

Covers the "Automatic Differentiation" and "Optimization" sections of
https://docs.pytorch.org/tutorials/beginner/basics/intro.html

The single most important idea in the whole framework: as you do arithmetic on tensors,
PyTorch records the operations in a graph. Calling .backward() on a scalar walks that
graph in reverse and computes the derivative of that scalar with respect to every input
that asked to be tracked. That is all "training" is.
"""

import torch


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


section("1. requires_grad turns on recording")

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2 * x
y.backward()
print(f"y = x^2 + 2x at x=3  ->  y = {y.item()}")
print(f"dy/dx = 2x + 2 = 8   ->  x.grad = {x.grad.item()}")
print(
    "\nNote that .grad lives on x, not on y. Gradients are always deposited on the\n"
    "*leaves* of the graph: the tensors you created, which in a real model are the weights."
)

print(f"\ny.requires_grad = {y.requires_grad}, y.grad_fn = {y.grad_fn}")
print("y knows the operation that produced it. That backward-pointing chain is the graph.")


section("2. Gradients ACCUMULATE. This is the #1 gotcha.")

w = torch.tensor(1.0, requires_grad=True)
for step in range(3):
    loss = w * 5
    loss.backward()
    print(f"  after backward #{step + 1}: w.grad = {w.grad.item()}")

print(
    "\nThe true derivative is 5 every time, but .grad went 5, 10, 15. PyTorch *adds* into\n"
    ".grad rather than overwriting it. That is deliberate: it lets you sum gradients from\n"
    "several losses, or accumulate over several mini-batches to fake a larger batch size."
)
w.grad = None
print("\nWhich is why every training loop calls optimizer.zero_grad() before backward().")
print("Forgetting it does not crash. It just quietly poisons training with stale gradients.")


section("3. The five-line training loop, in full")

torch.manual_seed(0)
true_w, true_b = 2.5, -1.0
xs = torch.linspace(-3, 3, 64).unsqueeze(1)
ys = true_w * xs + true_b + 0.05 * torch.randn_like(xs)

weight = torch.zeros(1, 1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([weight, bias], lr=0.05)

for epoch in range(201):
    prediction = xs @ weight + bias
    loss = ((prediction - ys) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"  epoch {epoch:>3}  loss {loss.item():.5f}  "
            f"w {weight.item():+.3f}  b {bias.item():+.3f}"
        )

print(f"\nrecovered w={weight.item():.3f} b={bias.item():.3f}   (truth: {true_w}, {true_b})")
print(
    """
Those five lines are the entire ritual, and they never change:

    prediction = model(inputs)          # forward: build the graph
    loss       = criterion(prediction, targets)
    optimizer.zero_grad()               # clear last step's gradients
    loss.backward()                     # reverse pass: fill every .grad
    optimizer.step()                    # nudge every parameter downhill

The seq2seq model in stage 3 has an encoder, a decoder, attention, and teacher forcing,
and its training loop is still exactly these five lines.
"""
)


section("4. Turning autograd off: no_grad and detach")

p = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    q = p * 10
print(f"inside no_grad: q.requires_grad = {q.requires_grad}")
print(
    "Use torch.no_grad() for evaluation and inference. It skips graph construction, which\n"
    "saves memory and time. Its absence during eval is why people mysteriously run out of\n"
    "memory while 'just testing'."
)

r = (p * 10).detach()
print(f"\ndetach() cuts one tensor out of the graph: r.requires_grad = {r.requires_grad}")
print(
    "detach() is how you stop a gradient from flowing backward through part of a model --\n"
    "for example, to stop a critic's gradient leaking into an actor in RL."
)


section("5. Non-scalar backward")

v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v**2
try:
    out.backward()
except RuntimeError as e:
    print(f"out.backward() on a non-scalar raises:\n  {e}\n")

out.backward(torch.ones_like(out))
print(f"You must say how to weight each output. With ones: v.grad = {v.grad}")
print(
    "\nThis is why losses are scalars. `.mean()` or `.sum()` at the end of your loss is not\n"
    "cosmetic -- it is what makes 'the' derivative well-defined."
)


section("6. Autograd is not only for neural networks")

theta = torch.tensor(0.7, requires_grad=True)
tip_height = torch.sin(theta) + 0.7 * torch.sin(2 * theta)
tip_height.backward()
print(f"d(height)/d(theta) at theta=0.7 rad = {theta.grad.item():.4f}")
print(
    """
Nothing above involved a model, a layer, or a dataset. Autograd differentiates *programs*.

That is the door into robotics: if you write your kinematics, your dynamics, or your
simulator in torch, you can take gradients through it and optimize with them. The next
script does exactly that -- it solves inverse kinematics with nothing but gradient descent.

Next: python robotics/stage1_ik_autograd.py
"""
)
