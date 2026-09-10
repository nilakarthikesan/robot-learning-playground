"""A first mechanistic-interpretability experiment on the 2-link arm.

Run:
    python robotics/stage1_interpret_fk.py

The forward-kinematics network learns joint angles -> fingertip position. We then ask:

1. Observation: can simple geometric quantities be decoded from its first hidden layer?
2. Intervention: which hidden units actually affect fingertip accuracy when ablated?

These are different questions. Correlation and linear probes show that information is
present, but not that the model uses it. Ablation supplies causal evidence, although even
that is imperfect because changing an activation can put the network off-distribution.
"""

from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arm2d import forward_kinematics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 7
HIDDEN = 64
TRAIN_SAMPLES = 6000
TEST_SAMPLES = 2000
EPOCHS = 50


class InterpretableFKNet(nn.Module):
    """Small MLP whose first hidden activation is easy to inspect and modify."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)

    def hidden1(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(theta))

    def forward(
        self,
        theta: torch.Tensor,
        ablate_unit: int | None = None,
        replacement: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h1 = self.hidden1(theta)
        if ablate_unit is not None:
            if replacement is None:
                raise ValueError("replacement is required when ablating a unit")
            h1 = h1.clone()
            h1[:, ablate_unit] = replacement[ablate_unit]
        h2 = torch.tanh(self.fc2(h1))
        return self.out(h2)


def sample_arm_data(n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    theta = (torch.rand(n_samples, 2, generator=generator) * 2 - 1) * torch.pi
    return theta, forward_kinematics(theta)


def train(model: nn.Module, theta: torch.Tensor, tip: torch.Tensor) -> None:
    loader = DataLoader(TensorDataset(theta, tip), batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for theta_batch, tip_batch in loader:
            prediction = model(theta_batch)
            loss = ((prediction - tip_batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * theta_batch.shape[0]

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:>2}: train MSE {total_loss / len(theta):.6f}")


def geometric_features(theta: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Known concepts that the network could use to solve forward kinematics."""
    t1, t2 = theta[:, 0], theta[:, 1]
    features = torch.stack(
        [
            torch.cos(t1),
            torch.sin(t1),
            torch.cos(t1 + t2),
            torch.sin(t1 + t2),
        ],
        dim=1,
    )
    names = ["cos(shoulder)", "sin(shoulder)", "cos(forearm)", "sin(forearm)"]
    return features, names


def correlation_matrix(activations: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Pearson correlations between every hidden unit and geometric feature."""
    activations = (activations - activations.mean(0)) / activations.std(0).clamp_min(1e-8)
    features = (features - features.mean(0)) / features.std(0).clamp_min(1e-8)
    return activations.T @ features / (activations.shape[0] - 1)


def linear_probe_r2(activations: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Fit a linear readout and report held-out R² for each known feature."""
    split = activations.shape[0] // 2
    train_x, test_x = activations[:split], activations[split:]
    train_y, test_y = features[:split], features[split:]

    train_x = torch.cat([train_x, torch.ones(len(train_x), 1)], dim=1)
    test_x = torch.cat([test_x, torch.ones(len(test_x), 1)], dim=1)
    weights = torch.linalg.lstsq(train_x, train_y).solution
    prediction = test_x @ weights
    residual = ((test_y - prediction) ** 2).sum(0)
    total = ((test_y - test_y.mean(0)) ** 2).sum(0)
    return 1 - residual / total


@torch.no_grad()
def ablation_effects(
    model: InterpretableFKNet,
    theta: torch.Tensor,
    tip: torch.Tensor,
    replacement: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Mean-replace one unit at a time and measure added fingertip error."""
    baseline = (model(theta) - tip).norm(dim=1).mean().item()
    ablated_errors = []
    for unit in range(HIDDEN):
        prediction = model(theta, ablate_unit=unit, replacement=replacement)
        ablated_errors.append((prediction - tip).norm(dim=1).mean())
    effects = torch.stack(ablated_errors) - baseline
    return baseline, effects


def plot_results(
    correlations: torch.Tensor,
    feature_names: list[str],
    effects: torch.Tensor,
    output_path: Path,
) -> None:
    selected = []
    for feature_index in range(correlations.shape[1]):
        for unit in correlations[:, feature_index].abs().topk(3).indices.tolist():
            if unit not in selected:
                selected.append(unit)
    shown_units = torch.tensor(selected)
    causal_units = effects.topk(10).indices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    image = ax1.imshow(
        correlations[shown_units].abs().T,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Observation: |correlation| with geometry")
    ax1.set_yticks(range(len(feature_names)), feature_names)
    ax1.set_xticks(range(len(shown_units)), [str(i.item()) for i in shown_units])
    ax1.set_xlabel("hidden unit")
    fig.colorbar(image, ax=ax1, label="absolute Pearson r")

    ax2.bar(range(len(causal_units)), effects[causal_units].cpu().numpy())
    ax2.set_title("Intervention: most damaging ablations")
    ax2.set_xticks(range(len(causal_units)), [str(i.item()) for i in causal_units])
    ax2.set_xlabel("hidden unit")
    ax2.set_ylabel("increase in mean fingertip error (m)")
    ax2.axhline(0, color="black", linewidth=0.8)

    fig.suptitle("What did the forward-kinematics network learn?")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    train_theta, train_tip = sample_arm_data(TRAIN_SAMPLES, seed=SEED)
    test_theta, test_tip = sample_arm_data(TEST_SAMPLES, seed=SEED + 1)

    print("Training angles -> fingertip network")
    model = InterpretableFKNet()
    train(model, train_theta, train_tip)
    model.eval()

    with torch.no_grad():
        train_activations = model.hidden1(train_theta)
        test_activations = model.hidden1(test_theta)
        features, feature_names = geometric_features(test_theta)
        correlations = correlation_matrix(test_activations, features)
        probe_scores = linear_probe_r2(test_activations, features)
        activation_means = train_activations.mean(0)
        baseline_error, effects = ablation_effects(
            model, test_theta, test_tip, activation_means
        )

    print(f"\nBaseline held-out fingertip error: {baseline_error:.4f} m")
    print("\nCan geometry be decoded from the hidden layer?")
    for index, name in enumerate(feature_names):
        unit = correlations[:, index].abs().argmax()
        corr = correlations[unit, index]
        print(
            f"  {name:<15} probe R²={probe_scores[index]:.3f}; "
            f"strongest unit={unit.item():>2}, r={corr:+.3f}"
        )

    print("\nWhich units causally affect accuracy when mean-ablated?")
    for unit in effects.topk(5).indices:
        print(f"  unit {unit.item():>2}: error increases by {effects[unit]:.4f} m")

    output_path = Path("out/stage1_interpretability.png")
    plot_results(correlations, feature_names, effects, output_path)
    print(f"\nSaved visualization to {output_path}")
    print(
        "\nInterpret carefully: a high probe score means information is decodable; "
        "it does not prove one neuron implements that concept. Ablation gives causal "
        "evidence, but hidden representations are distributed and redundant."
    )


if __name__ == "__main__":
    main()
