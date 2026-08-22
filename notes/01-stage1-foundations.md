# Stage 1 — Tensors, autograd, and the training loop

Source: [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

Code: `concepts/stage1_basics/01_tensors.py`, `concepts/stage1_basics/02_autograd.py`,
`robotics/stage1_ik_autograd.py`, `robotics/stage1_learn_fk.py`

---

## The one-paragraph version

A tensor is an n-dimensional array that also carries a dtype, a device, and optionally a
record of the operations that produced it. That record is a graph. Calling `.backward()` on
a scalar walks the graph in reverse and deposits, on every tracked input, the derivative of
that scalar with respect to it. An optimizer then nudges each of those inputs downhill.
Everything else in PyTorch — layers, datasets, transformers, robot policies — is
organization on top of those two mechanisms.

## Tensors

**Shape is the thing to track.** Read `(2, 3)` as "2 rows of 3", and remember the last
dimension is the one contiguous in memory. Almost every bug in stages 2 and 3 will be a
shape bug, and the habit that prevents them is printing shapes the moment you're unsure.

The operations that come up constantly:

| Op | Effect |
| --- | --- |
| `reshape(-1, k)` | reflow into k columns; `-1` means "infer" |
| `unsqueeze(d)` | insert a size-1 axis at position `d` |
| `squeeze()` | drop all size-1 axes |
| `transpose(a, b)` / `.T` | swap two axes |
| `@` / `matmul` | matrix multiply, batching over leading dims |
| `bmm` | same, for exactly 3D tensors — appears in attention |

`unsqueeze` and `squeeze` look like bureaucracy right now. They are load-bearing in
attention, where you line up a `(batch, 1, hidden)` query against `(batch, seq, hidden)`
keys and the size-1 axis is what makes the broadcast work.

**Broadcasting** aligns shapes from the right; each pair of dims must be equal or one must
be 1, and size-1 dims get stretched without copying. This is convenient and it is also how
you get silent bugs: `(3, 1) + (3,)` produces `(3, 3)` rather than raising. If a loss is
oddly shaped or a model trains but never improves, suspect broadcasting.

**Views share memory.** `reshape` and `transpose` usually give you a view onto the same
storage, so writing through one changes the other. Use `.clone()` when you want a copy.

**`dim=k` means "the axis that disappears."** `s.sum(dim=0)` on a `(2, 3)` gives `(3,)`.
Softmax over the wrong `dim` is a bug that produces plausible-looking garbage.

## Autograd

`requires_grad=True` turns on recording. Gradients land on the **leaves** of the graph —
the tensors you created, which in a model are the weights — never on intermediates.

Three things that are genuinely worth knowing, not just reading past:

**1. Gradients accumulate.** `.grad` is added into, not overwritten. Three `backward()`
calls on the same graph give `5, 10, 15` where the true derivative is `5`. This is
deliberate: it lets you sum several losses, or accumulate over mini-batches to simulate a
larger batch. It is also why `optimizer.zero_grad()` exists, and why forgetting it doesn't
crash — it just quietly poisons training.

**2. Losses must be scalars.** `.backward()` on a non-scalar raises "grad can be implicitly
created only for scalar outputs," because with multiple outputs there is no single
derivative. The `.mean()` at the end of your loss isn't cosmetic; it's what makes the
question well-posed.

**3. `no_grad` and `detach` are different tools.** `torch.no_grad()` is a context manager
that skips graph construction entirely — use it for evaluation, and its absence is why
people run out of memory "just testing." `detach()` cuts one specific tensor out of the
graph, which is how you stop gradient from flowing into part of a model.

## The loop that never changes

```python
prediction = model(inputs)
loss       = criterion(prediction, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The seq2seq model in stage 3 has an encoder, a decoder, attention, and teacher forcing, and
its training loop is still these five lines. Worth remembering when the architecture starts
looking intimidating.

## The three APIs that carry most PyTorch code

- **`nn.Module`** — anything you assign to `self` that's a Module or Parameter is
  auto-registered, which is how `.parameters()` and `.state_dict()` know what exists. That
  registration is the actual reason to subclass.
- **`Dataset`** — needs exactly `__len__` and `__getitem__`. That's the whole interface.
- **`DataLoader`** — batching, shuffling, parallel loading. `shuffle=True` on train (to
  decorrelate consecutive gradients), `False` on test.

Save the `state_dict`, not the model object; pickling the object bakes in your class
definition and breaks on refactor. Use `weights_only=True` when loading.

---

## What the robotics bridge actually showed

### Autograd differentiates programs, not just networks

`robotics/stage1_ik_autograd.py` solves inverse kinematics with no neural network at all.
Write the loss `‖forward_kinematics(θ) − target‖²` and run the same five-line ritual on θ
itself. It converges to machine precision.

Two by-products worth noticing:

- **The manipulator Jacobian comes for free and exactly.** `torch.autograd.functional.jacobian`
  on the forward-kinematics function matched the hand-derived Jacobian to `0.00e+00`.
  Autograd applies the chain rule op by op; it is not finite-differencing. That matrix
  (`v_tip = J θ̇`) is the workhorse of classical manipulator control.
- **Singularities are geometric, not numerical.** `det(J) = L₁L₂ sin θ₂`, so as the elbow
  straightens the condition number blew up from 2.6 to 483. Gradient descent doesn't crash
  there; it just gets very slow in one direction. Ill-conditioning presenting as
  "training is mysteriously slow" recurs everywhere.
- **A plateaued loss isn't automatically a bug.** Given an unreachable target, the optimizer
  extended the arm fully and pointed it straight at the goal — the best answer that exists.
  Residual error 1.30 m, irreducible. "Loss stopped going down" can mean bad LR, too-small
  model, *or* an infeasible objective, and those need opposite responses.

### The result that matters: multimodality breaks MSE

`robotics/stage1_learn_fk.py` runs one experiment three ways on identical data with an
identical 33k-parameter MLP:

| What we trained | Held-out task-space error |
| --- | --- |
| A: forward kinematics (θ → xy) | **0.025 m** (1.5% of reach) |
| B: inverse kinematics, joint-space MSE (xy → θ) | **0.941 m** (55% of reach) |
| C: inverse kinematics, task-space loss | **0.054 m** |

Part B is the interesting one. Nothing is wrong with the network, the data, or the
optimizer. The problem is in the *dataset*: for one hand position there are two correct
answers, elbow-up and elbow-down. Squared error is minimized by predicting the mean of the
targets you ask it to fit, so the network learns the average of the two — a configuration
that reaches neither. Its training loss even plateaued at 2.36 rather than going to zero,
which is the signature of irreducible target variance.

**The average of two good actions is usually a bad action.**

Part C fixes it without touching the network: instead of asking for specific angles, ask
that the resulting *position* be right, letting gradients flow back through the
differentiable forward kinematics. The loss no longer cares which valid solution the network
picks. 17× improvement from changing only the objective.

Two generalizable lessons:

1. A loss that goes down is not evidence a model works. Measure the thing you care about.
   Joint-space loss looked fine; task-space error was the honest metric.
2. When a model with enough capacity on enough data still fails, look at the loss before
   the architecture.

### Why this connects to real robot learning

This is the same structure as behavior cloning from demonstrations. A human teleoperator who
sometimes reaches left around an obstacle and sometimes right produces a dataset where MSE
training yields a policy that drives straight into the obstacle. A large fraction of the
current literature is attacking exactly the number in row B:

- **Diffusion Policy** — model the whole action distribution instead of a point estimate, so
  multiple modes can coexist.
- **ACT / action chunking** — predict a chunk of future actions, so the elbow-up/elbow-down
  choice is committed to once instead of re-litigated every timestep.
- **RT-2 / OpenVLA** — discretize actions into tokens and *classify*; a softmax over tokens
  can put mass on two separate modes, where a regression head cannot.

The Part C trick (differentiate through the physics) is the appeal of differentiable
simulation, and its limitation is right there too: our kinematics were smooth, and contact
and friction are not.

---

## Checkpoint questions

Worth being able to answer these without looking:

1. Why does `optimizer.zero_grad()` exist, and what breaks silently without it?
2. Why must the thing you call `.backward()` on be a scalar?
3. What's the difference between `no_grad()` and `detach()`, and when do you want each?
4. Part B's training loss plateaued at 2.36 instead of reaching 0. What does that number
   represent, and why couldn't more training or a bigger network reduce it?
5. Part C changed only the loss and got 17× better. Why did that work, and what property of
   the simulator did it require?
6. Give a manipulation task where MSE behavior cloning would fail for the Part B reason.
