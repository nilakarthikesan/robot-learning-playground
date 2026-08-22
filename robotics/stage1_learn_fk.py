"""Stage 1, second bridge: nn.Module, Dataset, DataLoader -- and a real failure.

Run me:  python robotics/stage1_learn_fk.py

This script covers the remaining pieces of "Learn the Basics" (building a model, datasets
and dataloaders, the optimization loop, saving and loading) on the same 2-link arm.

But it is arranged around an experiment, because the result is the point:

    Part A   train a network to learn FORWARD kinematics (angles -> position).  Works.
    Part B   train the same network to learn INVERSE kinematics (position -> angles),
             with the obvious mean-squared-error loss.                          Fails.
    Part C   fix Part B by changing the loss, not the network.                  Works.

Part B failing is not a bug to be tuned away. It is the multimodality problem from
stage1_ik_autograd.py, showing up as a number, and it is one of the central difficulties
in learning robot policies from demonstrations.
"""

import torch
import torch.nn as nn
from arm2d import forward_kinematics, workspace
from torch.utils.data import DataLoader, Dataset, random_split

torch.manual_seed(0)
DEVICE = "cpu"


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


class ArmKinematicsDataset(Dataset):
    """Random arm configurations paired with the resulting hand position.

    A Dataset needs exactly two methods: __len__ and __getitem__. That is the whole
    interface. Everything else (batching, shuffling, parallel loading) is DataLoader's job.
    """

    def __init__(self, n_samples=8192, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.theta = (torch.rand(n_samples, 2, generator=g) * 2 - 1) * torch.pi
        self.tip = forward_kinematics(self.theta)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):
        return self.theta[idx], self.tip[idx]


class MLP(nn.Module):
    """A plain multilayer perceptron.

    Anything you assign to `self` that is an nn.Module or nn.Parameter is automatically
    registered, which is how model.parameters() and model.state_dict() know what exists.
    That auto-registration is the actual reason to subclass nn.Module.
    """

    def __init__(self, in_dim=2, out_dim=2, hidden=128, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, loader, loss_fn, epochs=40, lr=1e-3, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for theta_batch, tip_batch in loader:
            loss = loss_fn(model, theta_batch, tip_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * theta_batch.shape[0]
            n += theta_batch.shape[0]
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    [{label}] epoch {epoch:>3}  train loss {total / n:.6f}")
    return model


section("0. Dataset and DataLoader")

full = ArmKinematicsDataset(n_samples=8192)
train_set, test_set = random_split(full, [7000, 1192], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

print(f"dataset size {len(full)}  ->  train {len(train_set)}  test {len(test_set)}")
theta_b, tip_b = next(iter(train_loader))
print(f"one batch: theta {tuple(theta_b.shape)}  tip {tuple(tip_b.shape)}")
print(
    """
The held-out test split is not a formality. The question "did it learn the relationship or
memorize the samples" is unanswerable without it, and in robotics the equivalent question
-- does the policy work on a configuration it never saw -- is the only one that matters.

shuffle=True on train and False on test is the standard pairing. Shuffling the training
set decorrelates consecutive gradients; shuffling the test set would only make your logs
harder to read.
"""
)


section("Part A: learn FORWARD kinematics (angles -> position)")


def fk_loss(model, theta, tip):
    return ((model(theta) - tip) ** 2).mean()


fk_net = MLP(in_dim=2, out_dim=2).to(DEVICE)
print(f"parameter count: {sum(p.numel() for p in fk_net.parameters()):,}\n")
train(fk_net, train_loader, fk_loss, epochs=40, label="FK")


@torch.no_grad()
def eval_fk(model, loader):
    model.eval()
    errs = []
    for theta, tip in loader:
        errs.append((model(theta) - tip).norm(dim=-1))
    return torch.cat(errs)


fk_err = eval_fk(fk_net, test_loader)
r_min, r_max = workspace()
print(f"\nheld-out position error: mean {fk_err.mean():.4f} m, median {fk_err.median():.4f} m, "
      f"worst {fk_err.max():.4f} m")
print(f"for scale, the arm's reach is {r_max:.2f} m, so mean error is "
      f"{100 * fk_err.mean() / r_max:.2f}% of reach")
print(
    """
It works, and it should: forward kinematics is a smooth function, one input maps to exactly
one output, and a network approximating a function is the case neural networks handle well.

(Note torch.no_grad() and model.eval() in the evaluator. no_grad skips graph construction.
model.eval() switches dropout and batchnorm into inference behavior -- this model has
neither, but calling it is a habit worth having, because the day you add batchnorm and
forget, your eval numbers will be wrong in a way that is very hard to spot.)
"""
)


section("Part B: learn INVERSE kinematics with the obvious loss (position -> angles)")


def ik_joint_loss(model, theta, tip):
    """Mean-squared error in JOINT space: 'predict the angles from the dataset'."""
    return ((model(tip) - theta) ** 2).mean()


ik_net = MLP(in_dim=2, out_dim=2).to(DEVICE)
train(ik_net, train_loader, ik_joint_loss, epochs=40, label="IK-joint")


@torch.no_grad()
def eval_ik(model, loader):
    """The honest metric: feed a target, take the predicted angles, and ask where the
    hand actually ends up. Joint-space loss is not what we care about."""
    model.eval()
    errs = []
    for theta, tip in loader:
        reached = forward_kinematics(model(tip))
        errs.append((reached - tip).norm(dim=-1))
    return torch.cat(errs)


ik_err = eval_ik(ik_net, test_loader)
print(f"\nheld-out TASK-SPACE error: mean {ik_err.mean():.4f} m, median {ik_err.median():.4f} m, "
      f"worst {ik_err.max():.4f} m")
print(f"that is {100 * ik_err.mean() / r_max:.1f}% of the arm's reach -- the hand is nowhere near "
      f"the target")
print(
    """
Look at what went wrong. The training loss went down. The network is the same size, the
data is the same data, the optimizer is the same optimizer. And the result is useless.

The cause: the dataset contains, for one hand position, two different correct answers
(elbow up and elbow down). Mean-squared error is minimized by predicting the *mean* of the
targets it is asked to fit. So the network dutifully learns the average of elbow-up and
elbow-down, which is a configuration that reaches neither.

This is the single most important idea in stage 1. Behavior cloning on real robot
demonstrations has the same structure: a human demonstrator who sometimes reaches around
the left side of an obstacle and sometimes the right side gives you a dataset where MSE
training produces a policy that drives straight into the obstacle. "The average of two
good actions is usually a bad action."

Modern answers to this include predicting a whole distribution rather than a point
(diffusion policies), predicting chunks of actions so the choice is made once rather than
re-litigated every timestep (action chunking), and discretizing actions into tokens and
classifying (RT-2, OpenVLA). All three are attacking the number printed above.
"""
)


section("Part C: same network, better loss -- differentiate through the physics")


def ik_task_loss(model, theta, tip):
    """Don't ask for specific angles. Ask that the resulting hand position be right.

    forward_kinematics is written in torch, so gradients flow through it back into the
    network. The loss no longer cares *which* of the valid solutions the network picks.
    """
    return ((forward_kinematics(model(tip)) - tip) ** 2).mean()


ik_net2 = MLP(in_dim=2, out_dim=2).to(DEVICE)
train(ik_net2, train_loader, ik_task_loss, epochs=40, label="IK-task")

ik_err2 = eval_ik(ik_net2, test_loader)
print(f"\nheld-out TASK-SPACE error: mean {ik_err2.mean():.4f} m, median {ik_err2.median():.4f} m, "
      f"worst {ik_err2.max():.4f} m")
print(f"\n  joint-space MSE loss : mean task error {ik_err.mean():.4f} m")
print(f"  task-space loss      : mean task error {ik_err2.mean():.4f} m")
improvement = ik_err.mean() / ik_err2.mean()
print(f"  -> {improvement:.1f}x better, with an identical network and identical data")
print(
    """
Nothing changed except what we asked for. The lesson generalizes well beyond arms: when a
model with enough capacity trained on enough data still fails, the loss function is the
first place to look, not the architecture.

The reason this fix was available at all is that our simulator is differentiable. That is
the appeal of differentiable simulation as a research direction, and also its limit --
contact and friction are not nearly this well behaved.
"""
)


section("Saving and loading")

import os

os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/stage1_fk_net.pt"
torch.save(fk_net.state_dict(), path)
print(f"saved state_dict to {path}")

reloaded = MLP(in_dim=2, out_dim=2)
reloaded.load_state_dict(torch.load(path, weights_only=True))
reloaded.eval()

probe = torch.tensor([[0.3, 0.8]])
print(f"\noriginal  net(probe) = {fk_net(probe).detach().tolist()}")
print(f"reloaded  net(probe) = {reloaded(probe).detach().tolist()}")
print(f"analytic  truth      = {forward_kinematics(probe).tolist()}")
print(
    """
Save the state_dict (a plain dict of tensors), not the model object. Pickling the object
bakes in your class definition and file paths, and it breaks the moment you refactor.
Constructing the model and loading weights into it is the portable way, and
weights_only=True refuses to execute arbitrary code from the file.
"""
)


section("What to take away")
print(
    """
  * nn.Module auto-registers parameters; Dataset needs only __len__ and __getitem__;
    DataLoader handles batching and shuffling. Those three cover most PyTorch code.
  * A loss that goes down is not evidence that a model works. Measure the thing you
    actually care about (here: task-space error, not joint-space error).
  * One-to-many problems break mean-squared-error regression. This is the reason a large
    part of the current robot learning literature exists.

Stage 1 is done. Next: stage 2, where inputs become sequences.
"""
)
