"""Export OmniReset (UR5e) rsl_rl checkpoints to SimDist policy/critic JITs.

Unlike ``export_policies.py`` (stock rsl_rl / Go2), this does NOT launch Isaac or
use the rsl_rl runner -- the UWLab rsl_rl fork's runner API is incompatible. The
checkpoints are plain MLP state_dicts, so we rebuild the actor/critic directly,
bake in the EmpiricalNormalization, and TorchScript them. Pure CPU torch.

For each ``model_{i}.pt`` under ``checkpoints/rl/<rl_run>/`` writes:
  * ``policies/policy_{i}.pt``  : obs["policy"] (215-d) -> action mean (7-d)
  * ``critics/critic_{i}.pt``   : obs["critic"] (204-d) -> value (1-d)

Both are ``torch.jit`` ScriptModules callable as ``module(obs)`` -- the contract
``simdist.utils.torch.get_actor_critic_from_iteration`` expects.

Usage:
    python scripts/export_policies_omnireset.py --rl_run omnireset_2026-06-21_17-10-39
"""

import argparse
import os
import re

import torch
import torch.nn as nn

from simdist.utils import paths


def find_model_files(directory):
    """Return [(path, iteration), ...] for model_<int>.pt under directory."""
    pattern = re.compile(r"model_(\d+)\.pt$")
    out = []
    for root, _, files in os.walk(directory):
        for f in files:
            m = pattern.match(f)
            if m:
                out.append((os.path.join(root, f), int(m.group(1))))
    return sorted(out, key=lambda x: x[1])


def build_mlp(state_dict: dict, prefix: str) -> nn.Sequential:
    """Rebuild an ELU MLP from rsl_rl ``<prefix>.<idx>.{weight,bias}`` keys.

    rsl_rl lays out Sequential as Linear(0) ELU(1) Linear(2) ELU(3) ... with no
    activation after the final Linear.
    """
    # Collect linear layer indices in order.
    idxs = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith(prefix + ".")}
    )
    layers: list[nn.Module] = []
    for n, i in enumerate(idxs):
        w = state_dict[f"{prefix}.{i}.weight"]
        b = state_dict[f"{prefix}.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if n < len(idxs) - 1:  # ELU between hidden layers, not after the output
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


# Must match rsl_rl (UWLab fork) EmpiricalNormalization.forward:
#   (x - _mean) / (_std + eps)   with eps = 1e-2
# Using raw _std would divide by zero on dims that were constant in training
# (e.g. fixed material/mass/joint props -> 71 zero-std dims in the critic) -> NaN.
NORM_EPS = 1e-2


class NormalizedModule(nn.Module):
    """EmpiricalNormalization then MLP: mlp((x - mean) / (std + eps))."""

    def __init__(self, mlp: nn.Sequential, mean: torch.Tensor, std: torch.Tensor, eps: float = NORM_EPS):
        super().__init__()
        self.mlp = mlp
        self.eps = eps
        # mean/std stored as (1, dim) in the checkpoint.
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((x - self.mean) / (self.std + self.eps))


def export_one(state_dict: dict, net_prefix: str, norm_prefix: str) -> torch.jit.ScriptModule:
    mlp = build_mlp(state_dict, net_prefix)
    mean = state_dict[f"{norm_prefix}._mean"]
    std = state_dict[f"{norm_prefix}._std"]
    module = NormalizedModule(mlp, mean, std).eval()
    return torch.jit.script(module)


def main():
    parser = argparse.ArgumentParser(description="Export OmniReset checkpoints for SimDist.")
    parser.add_argument(
        "-r", "--rl_run", type=str, required=True,
        help="Run folder name under checkpoints/rl/",
    )
    args = parser.parse_args()

    run_dir = paths.get_rl_run_dir(args.rl_run)
    policies_dir = paths.get_rl_policies_dir(args.rl_run)
    critics_dir = paths.get_rl_critics_dir(args.rl_run)
    os.makedirs(policies_dir, exist_ok=True)
    os.makedirs(critics_dir, exist_ok=True)

    models = find_model_files(run_dir)
    if not models:
        raise FileNotFoundError(f"No model_*.pt found under {run_dir}")
    print(f"Found {len(models)} checkpoints in {run_dir}")

    for path, it in models:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        policy = export_one(sd, "actor", "actor_obs_normalizer")
        critic = export_one(sd, "critic", "critic_obs_normalizer")
        policy.save(os.path.join(policies_dir, f"policy_{it}.pt"))
        critic.save(os.path.join(critics_dir, f"critic_{it}.pt"))
        print(f"  [{it:>4}] exported policy_{it}.pt + critic_{it}.pt")

    print(f"Done. policies -> {policies_dir}\n      critics  -> {critics_dir}")


if __name__ == "__main__":
    main()
