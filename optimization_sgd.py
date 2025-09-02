import argparse
from typing import Callable, List, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np


def f_quad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """f(x,y) = x^2 + y^2 (minimized at (0,0))."""
    return x * x + y * y


def f_neg_quad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """f(x,y) = -x^2 - y^2 (maximized at (0,0))."""
    return -(x * x + y * y)


@torch.no_grad()
def to_numpy_path(path_xy: List[Tuple[float, float]]) -> np.ndarray:
    return np.array(path_xy)


def optimize_path(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    start_xy: Tuple[float, float],
    steps: int = 50,
    lr: float = 0.2,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    maximize: bool = False,
    initial_momentum_mag: float = 0.0,
    dtype=torch.float32,
    device: str = "cpu",
) -> List[Tuple[float, float]]:
    """
    Run torch.optim.SGD on a single 2D parameter starting from start_xy and return the trajectory.
    """
    p = torch.nn.Parameter(torch.tensor(start_xy, dtype=dtype, device=device))
    opt = torch.optim.SGD([p], lr=lr, momentum=momentum, weight_decay=weight_decay, maximize=maximize)

    path: List[Tuple[float, float]] = []
    # record initial point
    path.append((float(p[0].detach().cpu()), float(p[1].detach().cpu())))

    if momentum > 0.0 and initial_momentum_mag > 0.0:
        # unit vector perpendicular to radius (orbital/tangential)
        with torch.no_grad():
            v = -p.detach()  # vector to origin
            norm = torch.norm(v)
            if norm > 0:
                # create perpendicular vector: rotate 90 degrees counterclockwise
                u = torch.stack([-v[1], v[0]]) / norm
                # scale by user knob; same dtype/device as param
                buf = (initial_momentum_mag * u).to(dtype=p.dtype, device=p.device)
                # set PyTorch SGD's internal momentum buffer
                state = opt.state[p]
                state["momentum_buffer"] = buf.clone()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = p[0], p[1]
        val = f(x, y)
        val.backward()
        opt.step()
        path.append((float(p[0].detach().cpu()), float(p[1].detach().cpu())))
    return path


def make_contour(
    ax: plt.Axes,
    f_np: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xlim=(-4, 4),
    ylim=(-4, 4),
    title: str = "",
):
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f_np(X, Y)
    # Use nicely spaced contour levels
    levels = np.geomspace(1e-3, Z.max() + 1e-3, 12) if Z.max() > 0 else np.linspace(Z.min(), -1e-3, 12)
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)


def plot_paths(
    ax: plt.Axes,
    paths: List[List[Tuple[float, float]]],
    labels: List[str],
    start_xy: Tuple[float, float],
):
    ax.scatter([start_xy[0]], [start_xy[1]], marker="o", s=40, label="start", zorder=5)
    ax.scatter([0], [0], marker="*", s=100, label="optimum (0,0)", zorder=6)
    for path, lab in zip(paths, labels):
        P = to_numpy_path(path)
        ax.plot(P[:, 0], P[:, 1], marker="o", markersize=2, linewidth=1.5, label=lab, alpha=0.6)
        # small arrows to show direction every few steps
        # skip = max(1, len(P) // 12)
        # for i in range(0, len(P) - skip, skip):
        #     dx, dy = P[i + skip, 0] - P[i, 0], P[i + skip, 1] - P[i, 1]
        #     ax.arrow(P[i, 0], P[i, 1], dx, dy, length_includes_head=True, head_width=0.08, head_length=0.12, alpha=0.4)
    ax.legend(loc="best", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Visualize SGD trajectories on simple 2D functions.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--start_x", type=float, default=-3.0)
    parser.add_argument("--start_y", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=2.0)
    parser.add_argument("--initial_momentum_mag", type=float, default=5.0)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    start_xy = (args.start_x, args.start_y)
    steps = args.steps
    lr = args.lr

    # ----- (a) Momentum sweep (weight_decay = 0) on f(x,y) = x^2 + y^2 -----
    momenta = [0.0, 0.5, 0.9]
    paths_a = [
        optimize_path(
            f=f_quad,
            start_xy=start_xy,
            steps=steps,
            lr=lr,
            momentum=m,
            weight_decay=0.0,
            maximize=False,
            initial_momentum_mag=args.initial_momentum_mag,
            device=device,
        )
        for m in momenta
    ]

    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=140)
    make_contour(ax1, f_np=lambda X, Y: X**2 + Y**2, title="Minimize $x^2+y^2$ (momentum sweep)")
    plot_paths(ax1, paths_a, [f"momentum={m}" for m in momenta], start_xy)
    fig1.tight_layout()
    fig1.savefig("plots/sgd_momentum.png")

    # ----- (b) Add weight decay on top of momentum sweep -----
    wd = float(args.weight_decay)
    paths_b = [
        optimize_path(
            f=f_quad,
            start_xy=start_xy,
            steps=steps,
            lr=lr,
            momentum=m,
            weight_decay=wd,
            maximize=False,
            initial_momentum_mag=args.initial_momentum_mag,
            device=device,
        )
        for m in momenta
    ]

    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=140)
    make_contour(ax2, f_np=lambda X, Y: X**2 + Y**2, title=f"Minimize $x^2+y^2$ (momentum sweep, weight_decay={wd})")
    plot_paths(ax2, paths_b, [f"momentum={m}, wd={wd}" for m in momenta], start_xy)
    fig2.tight_layout()
    fig2.savefig("plots/sgd_weight_decay.png")

    # ----- (c) Maximize -x^2 - y^2 (equivalently, ascend to (0,0)) -----
    paths_c = [
        optimize_path(
            f=f_neg_quad,
            start_xy=start_xy,
            steps=steps,
            lr=lr,
            momentum=m,
            weight_decay=0.0,
            maximize=True,  # key difference
            initial_momentum_mag=args.initial_momentum_mag,
            device=device,
        )
        for m in [0.0, 0.9]
    ]

    fig3, ax3 = plt.subplots(figsize=(6, 5), dpi=140)
    make_contour(ax3, f_np=lambda X, Y: -(X**2 + Y**2), title="Maximize $-x^2-y^2$ (maximize=True)")
    plot_paths(ax3, paths_c, ["momentum=0.0", "momentum=0.9"], start_xy)
    fig3.tight_layout()
    fig3.savefig("plots/sgd_maximize.png")

    # ----- Brief printed interpretation -----
    print("\n=== Interpretation ===")
    print("(a) Momentum: Higher momentum (e.g., 0.9) accelerates along consistent gradients, producing longer, more directed steps and potential overshoot/oscillation across level sets; momentum=0 is more direct but slower.")
    print(f"(b) Weight decay (wd={wd}): Adds an L2 pull toward the origin, effectively shrinking parameters each step. Trajectories dampen sooner and curve more tightly into (0,0), counteracting momentum overshoot.")
    print("(c) maximize=True on -x^2-y^2: Performs gradient ascent toward the maximum at (0,0). Behavior mirrors (a) qualitatively—momentum speeds approach and may overshoot before settling at the peak.")

    print("\nSaved figures: sgd_momentum.png, sgd_weight_decay.png, sgd_maximize.png")


if __name__ == "__main__":
    main()