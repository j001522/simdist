"""ResNet-18 image backbone (flax.nnx) with ImageNet weights ported from torchvision.

Used by ManipulationEncoder (paper app:manip / Table II): each of the 3 camera images
-> ResNet-18 -> 512-d global-avg-pooled feature. We hand-write the module in nnx (rather
than depend on flaxmodels/linen) so it lives natively in the simdist nnx model, is fully
controllable, and is cleanly swappable for a DINOv2 backbone later.

The 1000-way classification head is dropped; output is the 512-d pooled feature.

Weight port (torch -> flax layout):
  * Conv kernel: torch (O, I, kH, kW) -> flax (kH, kW, I, O)  == transpose(2, 3, 1, 0)
  * BatchNorm:   torch weight/bias/running_mean/running_var -> flax scale/bias/mean/var
torchvision resnet18 convs have no bias; BN epsilon = 1e-5.

Verify the port with scripts/verify_resnet_port.py (compares torch vs jax forward).
"""

import jax.numpy as jnp
from flax import linen as _nn  # only for max_pool
from flax import nnx

# ImageNet normalization (RGB, inputs in [0, 1]). NOTE(verify-on-data): confirm the
# decoded frames are RGB, not BGR -- cv2 round-trips consistently, so decoded == the
# array fed to imencode by the recorder; check its channel order once (port doc 3.0.0).
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

_BN_EPS = 1e-5


def imagenet_normalize(x: jnp.ndarray) -> jnp.ndarray:
    """(..., H, W, 3) uint8-or-[0,255] float -> ImageNet-normalized float32."""
    x = x.astype(jnp.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def _conv3x3(cin, cout, stride, rngs):
    return nnx.Conv(
        cin, cout, kernel_size=(3, 3), strides=(stride, stride),
        padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
    )


def _conv1x1(cin, cout, stride, rngs):
    return nnx.Conv(
        cin, cout, kernel_size=(1, 1), strides=(stride, stride),
        padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs,
    )


def _bn(features, rngs):
    return nnx.BatchNorm(features, epsilon=_BN_EPS, use_running_average=False, rngs=rngs)


class BasicBlock(nnx.Module):
    """torchvision BasicBlock (no bottleneck); optional 1x1 downsample on the skip."""

    def __init__(self, cin: int, cout: int, stride: int, downsample: bool, rngs: nnx.Rngs):
        self.conv1 = _conv3x3(cin, cout, stride, rngs)
        self.bn1 = _bn(cout, rngs)
        self.conv2 = _conv3x3(cout, cout, 1, rngs)
        self.bn2 = _bn(cout, rngs)
        if downsample:
            self.downsample_conv = _conv1x1(cin, cout, stride, rngs)
            self.downsample_bn = _bn(cout, rngs)
        else:
            self.downsample_conv = None
            self.downsample_bn = None

    def __call__(self, x, train: bool = False):
        ura = not train
        identity = x
        out = self.conv1(x)
        out = nnx.relu(self.bn1(out, use_running_average=ura))
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=ura)
        if self.downsample_conv is not None:
            identity = self.downsample_bn(
                self.downsample_conv(x), use_running_average=ura
            )
        return nnx.relu(out + identity)


class ResNet18Backbone(nnx.Module):
    """ResNet-18 feature extractor -> 512-d global-avg-pooled embedding per image.

    Input:  (..., H, W, 3) already ImageNet-normalized (call ``imagenet_normalize`` first,
            or set ``normalize=True``). Any leading batch dims are flattened internally.
    Output: (..., 512).
    """

    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            3, 64, kernel_size=(7, 7), strides=(2, 2),
            padding=((3, 3), (3, 3)), use_bias=False, rngs=rngs,
        )
        self.bn1 = _bn(64, rngs)
        # layer<k>: [BasicBlock(stride, downsample), BasicBlock(1, no downsample)]
        self.layer1 = nnx.List([
            BasicBlock(64, 64, 1, False, rngs), BasicBlock(64, 64, 1, False, rngs)])
        self.layer2 = nnx.List([
            BasicBlock(64, 128, 2, True, rngs), BasicBlock(128, 128, 1, False, rngs)])
        self.layer3 = nnx.List([
            BasicBlock(128, 256, 2, True, rngs), BasicBlock(256, 256, 1, False, rngs)])
        self.layer4 = nnx.List([
            BasicBlock(256, 512, 2, True, rngs), BasicBlock(512, 512, 1, False, rngs)])
        self.out_features = 512

    def __call__(self, x, train: bool = False, normalize: bool = False):
        # collapse leading dims -> (N, H, W, 3)
        lead = x.shape[:-3]
        x = x.reshape((-1,) + x.shape[-3:])
        if normalize:
            x = imagenet_normalize(x)
        ura = not train

        x = self.conv1(x)
        x = nnx.relu(self.bn1(x, use_running_average=ura))
        x = _nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                         padding=((1, 1), (1, 1)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x, train=train)
        x = jnp.mean(x, axis=(1, 2))  # global average pool -> (N, 512)
        return x.reshape(lead + (self.out_features,))


# --------------------------------------------------------------------------------------
# torchvision -> nnx weight porting
# --------------------------------------------------------------------------------------

def _copy_conv(nnx_conv: nnx.Conv, tv_conv) -> None:
    w = tv_conv.weight.detach().cpu().numpy()  # (O, I, kH, kW)
    nnx_conv.kernel.value = jnp.asarray(w.transpose(2, 3, 1, 0))
    assert tv_conv.bias is None, "torchvision resnet convs are bias-free"


def _copy_bn(nnx_bn: nnx.BatchNorm, tv_bn) -> None:
    nnx_bn.scale.value = jnp.asarray(tv_bn.weight.detach().cpu().numpy())
    nnx_bn.bias.value = jnp.asarray(tv_bn.bias.detach().cpu().numpy())
    nnx_bn.mean.value = jnp.asarray(tv_bn.running_mean.detach().cpu().numpy())
    nnx_bn.var.value = jnp.asarray(tv_bn.running_var.detach().cpu().numpy())


def load_torchvision_resnet18(model: ResNet18Backbone) -> ResNet18Backbone:
    """In-place copy of torchvision ImageNet-pretrained resnet18 weights into ``model``.

    Requires ``torch``/``torchvision`` (already in the training env for the dataloader).
    """
    import torchvision

    tv = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    tv.eval()

    _copy_conv(model.conv1, tv.conv1)
    _copy_bn(model.bn1, tv.bn1)
    for nnx_layer, tv_layer in (
        (model.layer1, tv.layer1),
        (model.layer2, tv.layer2),
        (model.layer3, tv.layer3),
        (model.layer4, tv.layer4),
    ):
        for nnx_block, tv_block in zip(nnx_layer, tv_layer):
            _copy_conv(nnx_block.conv1, tv_block.conv1)
            _copy_bn(nnx_block.bn1, tv_block.bn1)
            _copy_conv(nnx_block.conv2, tv_block.conv2)
            _copy_bn(nnx_block.bn2, tv_block.bn2)
            if nnx_block.downsample_conv is not None:
                _copy_conv(nnx_block.downsample_conv, tv_block.downsample[0])
                _copy_bn(nnx_block.downsample_bn, tv_block.downsample[1])
    return model
