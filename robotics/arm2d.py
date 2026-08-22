"""A two-link planar arm, written in PyTorch so that it is differentiable.

This is the simplest robot that is still interesting. Two joints, two links, moving in a
plane. Its forward kinematics are four lines of trigonometry, which means we can check every
answer by hand -- and because it is written with torch ops, autograd can differentiate
through it. That combination is what makes it a good teaching robot.

Conventions:
    theta[0]  angle of link 1 measured from the +x axis (radians)
    theta[1]  angle of link 2 measured *relative to link 1* (radians)

    All functions are batched: they accept a tensor of shape (..., 2) and return
    tensors whose leading dimensions match. That is the same convention PyTorch's own
    layers use, and getting comfortable with it now pays off in the sequence models later.
"""

import torch

L1 = 1.0
L2 = 0.7


def forward_kinematics(theta: torch.Tensor, l1: float = L1, l2: float = L2) -> torch.Tensor:
    """Joint angles -> end-effector (x, y).

    Args:
        theta: (..., 2) joint angles in radians.
    Returns:
        (..., 2) end-effector position.
    """
    t1 = theta[..., 0]
    t2 = theta[..., 1]
    x = l1 * torch.cos(t1) + l2 * torch.cos(t1 + t2)
    y = l1 * torch.sin(t1) + l2 * torch.sin(t1 + t2)
    return torch.stack((x, y), dim=-1)


def link_positions(theta: torch.Tensor, l1: float = L1, l2: float = L2) -> torch.Tensor:
    """Positions of all three points of the arm: base, elbow, end-effector.

    Useful for drawing. Returns (..., 3, 2).
    """
    t1 = theta[..., 0]
    t2 = theta[..., 1]
    base = torch.zeros_like(torch.stack((t1, t1), dim=-1))
    elbow = torch.stack((l1 * torch.cos(t1), l1 * torch.sin(t1)), dim=-1)
    tip = forward_kinematics(theta, l1, l2)
    return torch.stack((base, elbow, tip), dim=-2)


def analytic_jacobian(theta: torch.Tensor, l1: float = L1, l2: float = L2) -> torch.Tensor:
    """The manipulator Jacobian, derived by hand.

    J[i, j] = d(end-effector coordinate i) / d(joint angle j), shape (..., 2, 2).

    This exists so that `stage1_ik_autograd.py` can check autograd against calculus we
    did ourselves. In robotics this matrix is everywhere: it maps joint velocities to
    end-effector velocities, and where it loses rank the arm is at a singularity.
    """
    t1 = theta[..., 0]
    t2 = theta[..., 1]
    s1, c1 = torch.sin(t1), torch.cos(t1)
    s12, c12 = torch.sin(t1 + t2), torch.cos(t1 + t2)

    dx_dt1 = -l1 * s1 - l2 * s12
    dx_dt2 = -l2 * s12
    dy_dt1 = l1 * c1 + l2 * c12
    dy_dt2 = l2 * c12

    row_x = torch.stack((dx_dt1, dx_dt2), dim=-1)
    row_y = torch.stack((dy_dt1, dy_dt2), dim=-1)
    return torch.stack((row_x, row_y), dim=-2)


def workspace(l1: float = L1, l2: float = L2) -> tuple[float, float]:
    """Inner and outer radius of the reachable annulus."""
    return abs(l1 - l2), l1 + l2


def is_reachable(target: torch.Tensor, l1: float = L1, l2: float = L2) -> torch.Tensor:
    r_min, r_max = workspace(l1, l2)
    r = torch.linalg.norm(target, dim=-1)
    return (r >= r_min) & (r <= r_max)


def draw(thetas, target=None, path=None, title="2-link arm"):
    """Save a picture of one or more arm configurations. Returns the output path."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(thetas, torch.Tensor) and thetas.ndim == 1:
        thetas = thetas.unsqueeze(0)

    r_min, r_max = workspace()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Circle((0, 0), r_max, fill=False, ls="--", lw=0.8, color="0.7"))
    if r_min > 1e-9:
        ax.add_patch(plt.Circle((0, 0), r_min, fill=False, ls="--", lw=0.8, color="0.7"))

    pts = link_positions(thetas).detach()
    for i in range(pts.shape[0]):
        p = pts[i]
        ax.plot(p[:, 0], p[:, 1], "-o", lw=2.5, ms=6, label=f"solution {i + 1}")

    if target is not None:
        t = target.detach()
        ax.plot([t[0]], [t[1]], "x", ms=14, mew=3, color="crimson", label="target")

    lim = r_max * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

    path = path or "out/arm.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
