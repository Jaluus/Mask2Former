import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Tuple, Union

import os
import warnings
import urllib
import hashlib
from tqdm import tqdm
import numpy as np

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


model_urls = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride
        self.in_channels = inplanes
        self.out_channels = self.conv3.out_channels

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(Backbone):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, width=64):
        super().__init__()

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        # TODO: Get the Output shape of the stem with the avgpool
        out_channels_stem = width
        stride_stem = 4

        # fmt: off
        # residual layers, also called Stages
        # Layers are [3, 4, 6, 3] for R50
        # Each stage has 4 layers
        # We need the number of output channels per stage
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1, out_channels_1, stride_1 = self._make_layer(width, layers[0])
        self.layer2, out_channels_2, stride_2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3, out_channels_3, stride_3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4, out_channels_4, stride_4 = self._make_layer(width * 8, layers[3], stride=2)
        # fmt: on
        # TODO: Get the Output shape of the Stages#

        self.stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.stage_names = ["res2", "res3", "res4", "res5"]
        self._out_features = self.stage_names + ["stem"]
        self._out_feature_channels = {
            key: value
            for key, value in zip(
                self._out_features,
                [
                    out_channels_1,
                    out_channels_2,
                    out_channels_3,
                    out_channels_4,
                    out_channels_stem,
                ],
            )
        }
        self.stage_strides = {
            key: value
            for key, value in zip(
                self.stage_names, [stride_1, stride_2, stride_3, stride_4]
            )
        }

        self._out_feature_strides = {"stem": stride_stem}
        current_stride = stride_stem
        for name, stage in zip(self.stage_names, self.stages):
            self._out_feature_strides[name] = current_stride = (
                current_stride * self.stage_strides[name]
            )

        # This is initialized as an Identity layer so doesnt matter in the PODA Paper
        # embed_dim = width * 32
        # self.attnpool = AttentionPool2d(
        #     input_resolution // 32, embed_dim, heads, output_dim
        # )

    def _make_layer(self, planes, blocks, stride=1):

        # first layer
        layers = [Bottleneck(self._inplanes, planes, stride)]

        # Bottleneck.expansion = 4
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        feature_stride = np.prod([l.stride for l in layers])

        return nn.Sequential(*layers), layers[-1].out_channels, feature_stride

    def output_shape(self):
        # Need the Outchannels and the Stride of each Stage / Block and the Stem
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def forward_old(
        self,
        x,
    ):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        x = x.type(self.conv1.weight.dtype)

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        outputs = {}
        x = stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs


class CLIP_encoder(Backbone):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
    ):
        super().__init__()

        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
        )

        self.initialize_parameters()

    def initialize_parameters(self):

        for resnet_block in [
            self.visual.layer1,
            self.visual.layer2,
            self.visual.layer3,
            self.visual.layer4,
        ]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, x):
        return self.visual(x.type(self.dtype))


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@BACKBONE_REGISTRY.register()
def build_CLIP_backbone(cfg, input_shape):

    # vision_layers=(3, 4, 6, 3),
    # vision_width=64,
    # image_resolution=224,
    # embed_dim=1024,

    resnet_width = 64
    resnet_layers = (3, 4, 6, 3)

    model = ModifiedResNet(
        layers=resnet_layers,
        width=resnet_width,
    )

    # Load the Model
    with open("CLIP_state_dict.pt", "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    # freeze the Backbone
    for param in model.parameters():
        param.requires_grad = False

    return model


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


to_remove = [
    "positional_embedding",
    "text_projection",
    "logit_scale",
    "input_resolution",
    "context_length",
    "vocab_size",
    "transformer.resblocks.0.attn.in_proj_weight",
    "transformer.resblocks.0.attn.in_proj_bias",
    "transformer.resblocks.0.attn.out_proj.weight",
    "transformer.resblocks.0.attn.out_proj.bias",
    "transformer.resblocks.0.ln_1.weight",
    "transformer.resblocks.0.ln_1.bias",
    "transformer.resblocks.0.mlp.c_fc.weight",
    "transformer.resblocks.0.mlp.c_fc.bias",
    "transformer.resblocks.0.mlp.c_proj.weight",
    "transformer.resblocks.0.mlp.c_proj.bias",
    "transformer.resblocks.0.ln_2.weight",
    "transformer.resblocks.0.ln_2.bias",
    "transformer.resblocks.1.attn.in_proj_weight",
    "transformer.resblocks.1.attn.in_proj_bias",
    "transformer.resblocks.1.attn.out_proj.weight",
    "transformer.resblocks.1.attn.out_proj.bias",
    "transformer.resblocks.1.ln_1.weight",
    "transformer.resblocks.1.ln_1.bias",
    "transformer.resblocks.1.mlp.c_fc.weight",
    "transformer.resblocks.1.mlp.c_fc.bias",
    "transformer.resblocks.1.mlp.c_proj.weight",
    "transformer.resblocks.1.mlp.c_proj.bias",
    "transformer.resblocks.1.ln_2.weight",
    "transformer.resblocks.1.ln_2.bias",
    "transformer.resblocks.2.attn.in_proj_weight",
    "transformer.resblocks.2.attn.in_proj_bias",
    "transformer.resblocks.2.attn.out_proj.weight",
    "transformer.resblocks.2.attn.out_proj.bias",
    "transformer.resblocks.2.ln_1.weight",
    "transformer.resblocks.2.ln_1.bias",
    "transformer.resblocks.2.mlp.c_fc.weight",
    "transformer.resblocks.2.mlp.c_fc.bias",
    "transformer.resblocks.2.mlp.c_proj.weight",
    "transformer.resblocks.2.mlp.c_proj.bias",
    "transformer.resblocks.2.ln_2.weight",
    "transformer.resblocks.2.ln_2.bias",
    "transformer.resblocks.3.attn.in_proj_weight",
    "transformer.resblocks.3.attn.in_proj_bias",
    "transformer.resblocks.3.attn.out_proj.weight",
    "transformer.resblocks.3.attn.out_proj.bias",
    "transformer.resblocks.3.ln_1.weight",
    "transformer.resblocks.3.ln_1.bias",
    "transformer.resblocks.3.mlp.c_fc.weight",
    "transformer.resblocks.3.mlp.c_fc.bias",
    "transformer.resblocks.3.mlp.c_proj.weight",
    "transformer.resblocks.3.mlp.c_proj.bias",
    "transformer.resblocks.3.ln_2.weight",
    "transformer.resblocks.3.ln_2.bias",
    "transformer.resblocks.4.attn.in_proj_weight",
    "transformer.resblocks.4.attn.in_proj_bias",
    "transformer.resblocks.4.attn.out_proj.weight",
    "transformer.resblocks.4.attn.out_proj.bias",
    "transformer.resblocks.4.ln_1.weight",
    "transformer.resblocks.4.ln_1.bias",
    "transformer.resblocks.4.mlp.c_fc.weight",
    "transformer.resblocks.4.mlp.c_fc.bias",
    "transformer.resblocks.4.mlp.c_proj.weight",
    "transformer.resblocks.4.mlp.c_proj.bias",
    "transformer.resblocks.4.ln_2.weight",
    "transformer.resblocks.4.ln_2.bias",
    "transformer.resblocks.5.attn.in_proj_weight",
    "transformer.resblocks.5.attn.in_proj_bias",
    "transformer.resblocks.5.attn.out_proj.weight",
    "transformer.resblocks.5.attn.out_proj.bias",
    "transformer.resblocks.5.ln_1.weight",
    "transformer.resblocks.5.ln_1.bias",
    "transformer.resblocks.5.mlp.c_fc.weight",
    "transformer.resblocks.5.mlp.c_fc.bias",
    "transformer.resblocks.5.mlp.c_proj.weight",
    "transformer.resblocks.5.mlp.c_proj.bias",
    "transformer.resblocks.5.ln_2.weight",
    "transformer.resblocks.5.ln_2.bias",
    "transformer.resblocks.6.attn.in_proj_weight",
    "transformer.resblocks.6.attn.in_proj_bias",
    "transformer.resblocks.6.attn.out_proj.weight",
    "transformer.resblocks.6.attn.out_proj.bias",
    "transformer.resblocks.6.ln_1.weight",
    "transformer.resblocks.6.ln_1.bias",
    "transformer.resblocks.6.mlp.c_fc.weight",
    "transformer.resblocks.6.mlp.c_fc.bias",
    "transformer.resblocks.6.mlp.c_proj.weight",
    "transformer.resblocks.6.mlp.c_proj.bias",
    "transformer.resblocks.6.ln_2.weight",
    "transformer.resblocks.6.ln_2.bias",
    "transformer.resblocks.7.attn.in_proj_weight",
    "transformer.resblocks.7.attn.in_proj_bias",
    "transformer.resblocks.7.attn.out_proj.weight",
    "transformer.resblocks.7.attn.out_proj.bias",
    "transformer.resblocks.7.ln_1.weight",
    "transformer.resblocks.7.ln_1.bias",
    "transformer.resblocks.7.mlp.c_fc.weight",
    "transformer.resblocks.7.mlp.c_fc.bias",
    "transformer.resblocks.7.mlp.c_proj.weight",
    "transformer.resblocks.7.mlp.c_proj.bias",
    "transformer.resblocks.7.ln_2.weight",
    "transformer.resblocks.7.ln_2.bias",
    "transformer.resblocks.8.attn.in_proj_weight",
    "transformer.resblocks.8.attn.in_proj_bias",
    "transformer.resblocks.8.attn.out_proj.weight",
    "transformer.resblocks.8.attn.out_proj.bias",
    "transformer.resblocks.8.ln_1.weight",
    "transformer.resblocks.8.ln_1.bias",
    "transformer.resblocks.8.mlp.c_fc.weight",
    "transformer.resblocks.8.mlp.c_fc.bias",
    "transformer.resblocks.8.mlp.c_proj.weight",
    "transformer.resblocks.8.mlp.c_proj.bias",
    "transformer.resblocks.8.ln_2.weight",
    "transformer.resblocks.8.ln_2.bias",
    "transformer.resblocks.9.attn.in_proj_weight",
    "transformer.resblocks.9.attn.in_proj_bias",
    "transformer.resblocks.9.attn.out_proj.weight",
    "transformer.resblocks.9.attn.out_proj.bias",
    "transformer.resblocks.9.ln_1.weight",
    "transformer.resblocks.9.ln_1.bias",
    "transformer.resblocks.9.mlp.c_fc.weight",
    "transformer.resblocks.9.mlp.c_fc.bias",
    "transformer.resblocks.9.mlp.c_proj.weight",
    "transformer.resblocks.9.mlp.c_proj.bias",
    "transformer.resblocks.9.ln_2.weight",
    "transformer.resblocks.9.ln_2.bias",
    "transformer.resblocks.10.attn.in_proj_weight",
    "transformer.resblocks.10.attn.in_proj_bias",
    "transformer.resblocks.10.attn.out_proj.weight",
    "transformer.resblocks.10.attn.out_proj.bias",
    "transformer.resblocks.10.ln_1.weight",
    "transformer.resblocks.10.ln_1.bias",
    "transformer.resblocks.10.mlp.c_fc.weight",
    "transformer.resblocks.10.mlp.c_fc.bias",
    "transformer.resblocks.10.mlp.c_proj.weight",
    "transformer.resblocks.10.mlp.c_proj.bias",
    "transformer.resblocks.10.ln_2.weight",
    "transformer.resblocks.10.ln_2.bias",
    "transformer.resblocks.11.attn.in_proj_weight",
    "transformer.resblocks.11.attn.in_proj_bias",
    "transformer.resblocks.11.attn.out_proj.weight",
    "transformer.resblocks.11.attn.out_proj.bias",
    "transformer.resblocks.11.ln_1.weight",
    "transformer.resblocks.11.ln_1.bias",
    "transformer.resblocks.11.mlp.c_fc.weight",
    "transformer.resblocks.11.mlp.c_fc.bias",
    "transformer.resblocks.11.mlp.c_proj.weight",
    "transformer.resblocks.11.mlp.c_proj.bias",
    "transformer.resblocks.11.ln_2.weight",
    "transformer.resblocks.11.ln_2.bias",
    "token_embedding.weight",
    "ln_final.weight",
    "ln_final.bias",
]
#
def build_model(state_dict: dict):

    embed_dim = state_dict["text_projection"].shape[1]
    for k in to_remove:
        state_dict.pop(k, None)

    counts = [
        len(
            set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))
        )
        for b in [1, 2, 3, 4]
    ]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round(
        (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
    )
    assert (
        output_width**2 + 1
        == state_dict["visual.attnpool.positional_embedding"].shape[0]
    )
    image_resolution = output_width * 32

    model = CLIP_encoder(embed_dim, image_resolution, vision_layers, vision_width)

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
