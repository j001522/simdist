"""Debug-only pixel decoder: z -> camera images.

A port of DINO-WM's ``models/decoder/transposed_conv.py``
(github.com/gaoyuezhou/dino_wm) from PyTorch to flax nnx, which is itself the
DreamerV2/V3 transposed-conv decoder. Chosen over inventing one because it takes a
FLAT embedding vector as input -- which is exactly our fused ``z`` -- and reaches full
input resolution natively. LeWM describes an equivalent CLS-token decoder but never
released the code (github.com/lucas-maes/le-wm ships no decoder).

NOTHING here feeds back into the world model. The decoder is trained post-hoc on a
frozen checkpoint and its gradient never reaches the encoder, dynamics or DINOv2 --
see decoder_trainer.py, where z is computed in a separate jitted function and handed
to the loss as a constant. This is deliberate, not incidental: DINO-WM detaches for
the same reason (``z.detach()  # recon loss should only affect decoder``), and LeWM's
own ablation is that a reconstruction term inside the training loop costs control
performance (96.0 -> 86.0 SR).

What the reconstruction can and cannot show: pooling deliberately discards the patch
grid (one vector per camera, no spatial tokens) and z is a further latent_dim
bottleneck over all cameras plus proprio. Expect arm pose and object position to
survive and fine texture not to. Blurry output is the expected result, not a bug.

Multi-camera (``head_mode``): the manipulation system has n_cam=3 views but a single
fused z. Decoding every view from that one vector is the point -- it is what z actually
has to carry -- but there are two ways to spend parameters on it.

``shared`` (the original, and what every decoder trained before 2026-08-18 used) runs
ONE stack and emits ``n_cam * channels`` maps from the last layer, split into per-camera
images. At depth 64 that puts 31.06M parameters in the stack and 14.4k -- 0.05% -- in
the only layer that knows which camera it is drawing. Three viewpoints have to be
superimposed in one set of feature maps at the same spatial locations and separated
per-pixel by a single 5x5 kernel. Measured consequence: the wrist view (the
highest-variance camera, ~40% of the gradient) comes out legible while front and side
come out as blobs, and quadrupling the trunk (depth 32 -> 64) barely moved it.

``split`` runs a full independent stack per camera -- n_cam x the parameters and n_cam x
the decode cost, no superposition anywhere. This is the standard shape in the multi-view
literature: MV-MWM conditions a shared decoder on per-view embeddings, MultiWorld
decomposes multi-view generation into per-view generators over a shared state.
"""

from typing import List

import flax.nnx as nnx
import jax
import jax.numpy as jnp


# DINO-WM's decoder is a fixed 5-layer stack. With stride 3 / kernel 5 / pad 1 each
# layer triples the spatial size ((s-1)*3 + 5 - 2 == 3s), so the stack runs
# 1 -> 3 -> 9 -> 27 -> 81 -> 243 and a final resize lands on the target resolution.
_NUM_LAYERS = 5


class TransposedConvDecoder(nnx.Module):
    """z -> (..., n_cam, image_size, image_size, channels).

    Args mirror DINO-WM's ``TransposedConvDecoder``: ``depth`` scales every channel
    count, ``kernel_size``/``stride`` shape the upsampling schedule. Their config
    (``conf/decoder/transposed_conv.yaml``) uses depth=64, kernel=5, stride=3 at
    224x224; depth=32 is the cheaper default here since this only ever runs as a probe.
    ``head_mode`` is ours and is described in the module docstring: ``shared`` is one
    stack for all cameras, ``split`` is one stack each.

    Output is linear (no sigmoid/tanh), as in DINO-WM, where it is the mean of a unit-
    variance Normal and therefore trained under plain MSE. Targets are images in
    [0, 1]; clip at visualization time, not here.
    """

    def __init__(
        self,
        latent_dim: int,
        n_cam: int,
        image_size: int,
        channels: int = 3,
        depth: int = 32,
        kernel_size: int = 5,
        stride: int = 3,
        head_mode: str = "shared",
        rngs: nnx.Rngs = None,
    ):
        self.latent_dim = latent_dim
        self.n_cam = n_cam
        self.image_size = image_size
        self.channels = channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        if head_mode not in ("shared", "split"):
            raise ValueError(f"unknown head_mode {head_mode!r} (shared|split)")
        self.head_mode = head_mode
        # PyTorch's ConvTranspose2d(padding=p) shrinks the output by 2p relative to the
        # unpadded result. flax's ConvTranspose has no such argument, so we run 'VALID'
        # and crop p from each side -- the same arithmetic, spelled out.
        self.crop = (kernel_size - stride) // 2 if kernel_size > stride else 0

        # depth*32 -> depth*8 -> depth*4 -> depth*2 -> depth*1 -> out
        self.in_channels = depth * 32
        trunk: List[int] = [depth * 8, depth * 4, depth * 2, depth * 1]

        # kaiming/he uniform + zero bias, matching DINO-WM's initialize_weights().
        kernel_init = nnx.initializers.he_uniform()

        def make_proj():
            return nnx.Linear(
                latent_dim, self.in_channels, kernel_init=kernel_init, rngs=rngs
            )

        def make_layers(out_channels: int):
            layers = nnx.List([])
            prev = self.in_channels
            for f in trunk + [out_channels]:
                layers.append(
                    nnx.ConvTranspose(
                        in_features=prev,
                        out_features=f,
                        kernel_size=(kernel_size, kernel_size),
                        strides=(stride, stride),
                        padding="VALID",
                        kernel_init=kernel_init,
                        rngs=rngs,
                    )
                )
                prev = f
            assert len(layers) == _NUM_LAYERS
            return layers

        if head_mode == "shared":
            # One stack for every view; the cameras exist only as a channel split in
            # the last layer. Attribute names are load-bearing -- they are the paths
            # the pre-2026-08-18 checkpoints were saved under, so leave them alone.
            self.proj = make_proj()
            self.layers = make_layers(n_cam * channels)
        else:
            # One full stack per camera. Different attribute names on purpose: a split
            # decoder is a different state layout and must not silently restore from a
            # shared checkpoint.
            self.cam_proj = nnx.List([make_proj() for _ in range(n_cam)])
            self.cam_layers = nnx.List([make_layers(channels) for _ in range(n_cam)])

    @property
    def pre_resize_size(self) -> int:
        """Spatial size the transposed-conv stack reaches before the final resize."""
        s = 1
        for _ in range(_NUM_LAYERS):
            s = (s - 1) * self.stride + self.kernel_size - 2 * self.crop
        return s

    def _stack(self, proj, layers, x: jnp.ndarray) -> jnp.ndarray:
        """(N, latent_dim) -> (N, image_size, image_size, out_channels)."""
        n = x.shape[0]
        h = proj(x).reshape(n, 1, 1, self.in_channels)  # NHWC

        last = len(layers) - 1
        for i, layer in enumerate(layers):
            h = layer(h)
            if self.crop:
                h = h[:, self.crop : -self.crop, self.crop : -self.crop, :]
            if i != last:
                h = nnx.relu(h)

        if h.shape[1] != self.image_size:
            h = jax.image.resize(
                h,
                (n, self.image_size, self.image_size, h.shape[-1]),
                method="bilinear",
            )
        return h

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """``z``: (..., latent_dim). Returns (..., n_cam, image_size, image_size, C).

        Leading dimensions are flattened and restored, so the same call decodes a single
        latent (B, D) or a stack of them (B, T, D) -- but note that every latent is
        drawn INDEPENDENTLY. There is no time axis inside the decoder and it cannot tell
        which step, if any, a latent came from.
        """
        lead = z.shape[:-1]
        x = z.reshape(-1, self.latent_dim)
        n = x.shape[0]

        if self.head_mode == "shared":
            x = self._stack(self.proj, self.layers, x)
            # Split the channel axis into (camera, colour). Which half-open slice
            # belongs to which camera is a convention the decoder simply learns; keep
            # it stable so a trained decoder stays readable.
            x = x.reshape(n, self.image_size, self.image_size, self.n_cam, self.channels)
            x = jnp.transpose(x, (0, 3, 1, 2, 4))  # (N, n_cam, H, W, C)
        else:
            x = jnp.stack(
                [
                    self._stack(self.cam_proj[c], self.cam_layers[c], x)
                    for c in range(self.n_cam)
                ],
                axis=1,
            )  # (N, n_cam, H, W, C)

        return x.reshape(lead + x.shape[1:])


def get_decoder(
    decoder_cfg: dict,
    latent_dim: int,
    n_cam: int,
    image_size: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Build the decoder named by ``decoder_cfg['type']``."""
    dec_type = decoder_cfg.get("type", "transposed_conv")
    if dec_type != "transposed_conv":
        raise ValueError(f"unknown decoder type {dec_type!r}")
    return TransposedConvDecoder(
        latent_dim=latent_dim,
        n_cam=n_cam,
        image_size=image_size,
        channels=decoder_cfg.get("channels", 3),
        depth=decoder_cfg.get("depth", 32),
        kernel_size=decoder_cfg.get("kernel_size", 5),
        stride=decoder_cfg.get("stride", 3),
        # Absent in configs written before 2026-08-18; those runs are all shared.
        head_mode=decoder_cfg.get("head_mode", "shared"),
        rngs=rngs,
    )
