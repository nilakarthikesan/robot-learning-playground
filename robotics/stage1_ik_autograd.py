"""Stage 1, robotics bridge: solve inverse kinematics with autograd.

Run me:  python robotics/stage1_ik_autograd.py

Forward kinematics is easy: given joint angles, where is the hand? Four lines of trig.
Inverse kinematics is the useful one: given a point in space, what joint angles put the
hand there? For a 2-link arm you can solve it in closed form with the law of cosines, but
for a 7-joint arm it gets ugly, and for a humanoid it is hopeless.

So we do what deep learning does to everything: write down how wrong we are, and walk
downhill.

    loss(theta) = || forward_kinematics(theta) - target ||^2

and let autograd hand us d(loss)/d(theta). No neural network anywhere -- just the same
five-line optimization ritual from 02_autograd.py, applied to a robot.

Three things in here matter more than the IK itself, because all three come back when you
get to real policies:
  * local minima are a property of the problem, not a bug in the optimizer
  * autograd computes the manipulator Jacobian for free, and correctly
  * near a singularity the geometry, not the code, is what breaks
"""

import math

import torch
from arm2d import (
    L1,
    L2,
    analytic_jacobian,
    draw,
    forward_kinematics,
    is_reachable,
    link_positions,
    workspace,
)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def solve_ik(target, theta_init, steps=400, lr=0.05, verbose_every=100):
    """Gradient descent on squared distance to the target. Returns (theta, history)."""
    theta = theta_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    history = []

    for step in range(steps):
        tip = forward_kinematics(theta)
        loss = ((tip - target) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if verbose_every and step % verbose_every == 0:
            print(
                f"    step {step:>4}  loss {loss.item():.6f}  "
                f"theta = ({theta[0].item():+.3f}, {theta[1].item():+.3f})  "
                f"grad = ({theta.grad[0].item():+.4f}, {theta.grad[1].item():+.4f})"
            )

    return theta.detach(), history


section("0. Sanity-check the forward kinematics by hand")

print(f"link lengths: L1 = {L1}, L2 = {L2}")
straight = torch.tensor([0.0, 0.0])
print(f"\ntheta = (0, 0)      -> tip = {forward_kinematics(straight).tolist()}")
print(f"   expected ({L1 + L2}, 0.0): arm fully extended along +x. ok")

folded = torch.tensor([0.0, math.pi])
print(f"\ntheta = (0, pi)     -> tip = {forward_kinematics(folded).tolist()}")
print(f"   expected ({L1 - L2:.1f}, ~0.0): link 2 folded back on link 1. ok")

up = torch.tensor([math.pi / 2, 0.0])
print(f"\ntheta = (pi/2, 0)   -> tip = {forward_kinematics(up).tolist()}")
print(f"   expected (~0.0, {L1 + L2}): straight up. ok")

r_min, r_max = workspace()
print(f"\nreachable set is the annulus {r_min:.2f} <= r <= {r_max:.2f}")
print("Never trust kinematics code you have not checked against a configuration you can")
print("picture in your head. This takes thirty seconds and saves hours.")


section("1. IK by gradient descent")

target = torch.tensor([1.2, 0.6])
print(f"target = {target.tolist()}, |target| = {target.norm():.3f}, reachable = {bool(is_reachable(target))}\n")

theta_a, hist_a = solve_ik(target, torch.tensor([0.1, 0.1]))
tip_a = forward_kinematics(theta_a)
print(f"\n  converged theta = ({theta_a[0].item():+.4f}, {theta_a[1].item():+.4f}) rad")
print(f"  reached tip     = {[round(v, 5) for v in tip_a.tolist()]}")
print(f"  final error     = {(tip_a - target).norm().item():.2e} m")

print(
    "\nWe never told it anything about arms, angles, or trigonometry inverses. We described\n"
    "the goal, and autograd supplied the direction."
)


section("2. Local minima: same target, different starting guess")

theta_b, hist_b = solve_ik(target, torch.tensor([1.4, -0.1]), verbose_every=0)
tip_b = forward_kinematics(theta_b)

print(f"start (0.10, 0.10) -> theta = ({theta_a[0].item():+.4f}, {theta_a[1].item():+.4f})"
      f"   error {(tip_a - target).norm().item():.2e}")
print(f"start (1.40, -0.10) -> theta = ({theta_b[0].item():+.4f}, {theta_b[1].item():+.4f})"
      f"   error {(tip_b - target).norm().item():.2e}")

elbow_a = link_positions(theta_a)[1]
elbow_b = link_positions(theta_b)[1]
print(f"\nBoth reach the target, but the elbow is in a different place:")
print(f"  solution 1 elbow at {[round(v, 3) for v in elbow_a.tolist()]}")
print(f"  solution 2 elbow at {[round(v, 3) for v in elbow_b.tolist()]}")
print(
    """
This is "elbow up" versus "elbow down", and it is the whole story of multimodality in one
picture. The loss has two equally good global minima. Gradient descent finds whichever one
it started nearest, and it cannot tell you the other exists.

Why you should care: this is *exactly* the failure that motivates diffusion policies and
action chunking in robot learning. If two different actions are both correct, a network
trained with mean-squared error will learn to predict their average -- and the average of
elbow-up and elbow-down is an elbow position that hits the table. Averaging two valid
behaviors usually gives you an invalid one.
"""
)

path = draw(
    torch.stack([theta_a, theta_b]),
    target=target,
    path="out/stage1_ik_two_solutions.png",
    title="Two IK solutions for one target",
)
print(f"picture written to {path}")


section("3. The manipulator Jacobian, for free")

theta_probe = torch.tensor([0.6, 0.9])
J_auto = torch.autograd.functional.jacobian(forward_kinematics, theta_probe)
J_hand = analytic_jacobian(theta_probe)

print(f"autograd Jacobian:\n{J_auto}")
print(f"\nhand-derived Jacobian:\n{J_hand}")
print(f"\nmax absolute difference: {(J_auto - J_hand).abs().max().item():.2e}")
print(
    """
Identical to floating-point noise. autograd is not approximating the derivative with
finite differences; it is applying the chain rule symbolically, op by op, and getting
the exact answer.

J is the matrix that maps joint velocities to hand velocities: v_tip = J @ theta_dot.
It is the most-used object in classical manipulator control, and you get it from a
one-line call on code you wrote for a different purpose.
"""
)


section("4. Singularities: where the geometry, not the code, breaks down")

print("det(J) = L1 * L2 * sin(theta2), so it vanishes when the arm is straight.\n")
for t2 in [1.5, 0.5, 0.1, 0.01]:
    th = torch.tensor([0.3, t2])
    J = analytic_jacobian(th)
    det = torch.linalg.det(J)
    cond = torch.linalg.cond(J)
    print(f"  theta2 = {t2:>5}  det(J) = {det.item():+.5f}   condition number = {cond.item():>10.1f}")

print(
    """
As the elbow straightens, J becomes singular. Physically: with a straight arm you cannot
move the hand radially at all, no matter how you spin the joints -- you have lost a
direction of motion. Classical control handles this with damped least squares; learned
policies handle it by never being told about it and hopefully seeing enough data.

Notice that gradient descent does not crash here, it just becomes very slow in one
direction. Ill-conditioning showing up as "training is mysteriously slow" is a theme you
will meet again.
"""
)


section("5. An unreachable target")

far = torch.tensor([2.4, 1.8])
print(f"target = {far.tolist()}, |target| = {far.norm():.3f} > r_max = {r_max:.3f}")
print(f"reachable = {bool(is_reachable(far))}\n")

theta_far, hist_far = solve_ik(far, torch.tensor([0.2, 0.6]), verbose_every=0)
tip_far = forward_kinematics(theta_far)
print(f"converged theta = ({theta_far[0].item():+.4f}, {theta_far[1].item():+.4f})")
print(f"reached tip     = {[round(v, 4) for v in tip_far.tolist()]}, |tip| = {tip_far.norm():.4f}")
print(f"residual error  = {(tip_far - far).norm().item():.4f} m  (irreducible)")
print(f"elbow angle theta2 = {theta_far[1].item():+.4f} rad -> essentially straight")

print(
    """
The optimizer did the most sensible possible thing: it stretched the arm out fully and
pointed it directly at the target. The loss plateaus at a nonzero value because there is
no theta that achieves zero.

The lesson is about diagnosis. "Loss went down and then stopped at 0.4" can mean your
learning rate is wrong, or your model is too small -- or it can mean the task is
infeasible and the model has already found the best answer that exists. Those require
opposite responses, and only knowing your problem tells them apart.
"""
)

path = draw(
    theta_far.unsqueeze(0),
    target=far,
    path="out/stage1_ik_unreachable.png",
    title="Unreachable target: arm extends and points",
)
print(f"picture written to {path}")


section("What to take away")
print(
    """
  * Autograd differentiates any torch program, not just neural networks. Write your
    physics in torch and you can optimize through it.
  * Multiple valid solutions to one goal is normal in robotics, and squared-error
    training averages them into something wrong. Remember this in stage 3.
  * A plateaued loss is not automatically a bug.

Next: python robotics/stage1_learn_fk.py -- the same arm, but now with an actual network,
a Dataset, and a DataLoader, which are the pieces stage 2 is built from.
"""
)
