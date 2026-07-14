"""Verify the nnx ResNet-18 port matches torchvision within tolerance.

Run once in the simdist-jax env (needs torch + torchvision + jax):
    python scripts/verify_resnet_port.py

Builds the nnx backbone, ports ImageNet weights from torchvision, runs both on the same
random image in EVAL mode (BN running stats), and compares the 512-d pooled features.
torchvision resnet18 features == output of global avgpool (before fc). A small L2/max
error (<~1e-3) confirms the layout mapping (conv transpose, BN params, padding) is right.
"""

import numpy as np


def main():
    import jax.numpy as jnp
    import torch
    import torchvision
    from flax import nnx

    from simdist.modeling.resnet import (
        ResNet18Backbone,
        imagenet_normalize,
        load_torchvision_resnet18,
    )

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(2, 224, 224, 3), dtype=np.uint8)  # NHWC uint8

    # --- jax ---
    model = ResNet18Backbone(rngs=nnx.Rngs(0))
    load_torchvision_resnet18(model)
    model.eval()
    x_jax = imagenet_normalize(jnp.asarray(img))
    feat_jax = np.asarray(model(x_jax, train=False))  # (2, 512)

    # --- torch (feature extractor = everything up to and incl. avgpool) ---
    tv = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_t = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0  # NCHW
    x_t = (x_t - mean) / std
    with torch.no_grad():
        f = tv.conv1(x_t); f = tv.bn1(f); f = tv.relu(f); f = tv.maxpool(f)
        f = tv.layer1(f); f = tv.layer2(f); f = tv.layer3(f); f = tv.layer4(f)
        f = tv.avgpool(f).flatten(1)  # (2, 512)
    feat_torch = f.numpy()

    max_err = np.abs(feat_jax - feat_torch).max()
    l2 = np.linalg.norm(feat_jax - feat_torch) / np.linalg.norm(feat_torch)
    print(f"jax  feat: shape={feat_jax.shape} mean={feat_jax.mean():.4f}")
    print(f"torch feat: shape={feat_torch.shape} mean={feat_torch.mean():.4f}")
    print(f"max abs err = {max_err:.2e}   rel L2 = {l2:.2e}")
    ok = max_err < 1e-2 and l2 < 1e-3
    print("PORT OK" if ok else "PORT MISMATCH -- check conv transpose / BN / padding")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
