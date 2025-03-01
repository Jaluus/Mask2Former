# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

import clip
import numpy as np


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder_CLIP_TEXTPERTURB(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        # Custom Variables
        self.isInference = False
        # End Custom Variables

        self.cityscapes_classes = [
            [
                "road",
                "Street",
                "Highway",
                "Lane",
                "Pathway",
                "Avenue",
                "Boulevard",
                "Route",
                "Thoroughfare",
                "Byway",
                "Motorway",
            ],
            [
                "sidewalk",
                "Footpath",
                "Pavement",
                "Pathway",
                "Walkway",
                "Sidetrack",
                "Trail",
                "Boardwalk",
                "Promenade",
                "Esplanade",
                "Pedestrian Way",
            ],
            [
                "Building",
                "Edifice",
                "Structure",
                "Construction",
                "Premises",
                "Dwelling",
                "House",
                "Shelter",
                "Facility",
                "Tower",
                "Skyscraper",
            ],
            [
                "wall",
                "Barrier",
                "Partition",
                "Divider",
                "Enclosure",
                "Panel",
                "Shield",
                "Bulkhead",
                "Barricade",
                "Dam",
                "Embankment",
            ],
            [
                "fence",
                "Enclosure",
                "Hedge",
                "Palisade",
                "Perimeter",
                "Stockade",
                "Windbreak",
                "Picket",
                "Railing",
                "Guardrail",
            ],
            [
                "pole",
                "Post",
                "Column",
                "Pillar",
                "Stanchion",
                "Mast",
                "Staff",
                "Standard",
                "Rod",
                "Spire",
                "Perch",
            ],
            [
                "traffic light",
                "Stoplight",
                "Traffic Signal",
                "Semaphore",
                "Red Light",
                "Green Light",
                "Intersection Light",
                "Traffic Control Device",
                "Road Signal",
                "Traffic Beacon",
                "Traffic Lamp",
            ],
            [
                "traffic sign",
                "Road Sign",
                "Traffic Signal",
                "Directional Sign",
                "Warning Sign",
                "Information Sign",
                "Regulatory Sign",
                "Guide Sign",
                "Street Sign",
                "Speed Limit Sign",
                "No Parking Sign",
            ],
            [
                "vegetation",
                "Flora",
                "Greenery",
                "Trees",
                "Foliage",
                "Undergrowth",
                "Shrubs",
                "Woodland",
                "Forest",
                "Jungle",
                "Thicket",
            ],
            [
                "terrain",
                "Landscape",
                "Ground",
                "Surface",
                "Soil",
                "Earth",
                "Land",
                "Grounds",
                "Groundwork",
                "Ground Cover",
            ],
            ["sky", "Atmosphere", "Heavens", "Air", "Celestial Sphere", "Firmament"],
            [
                "person",
                "Pedestrian",
                "Walker",
                "Stroller",
                "Hiker",
                "Traveler",
                "Pilgrim",
                "Hitchhiker",
                "Commute",
                "Wayfarer",
                "Foot Passenger",
                "Passenger",
            ],
            [
                "rider",
                "Cyclist",
                "Biker",
                "Racer",
                "Rider",
                "Pedaler",
                "Bicycle User",
                "Two-wheeler",
                "Pedal Pusher",
                "Bike Commuter",
                "Bike Rider",
            ],
            [
                "car",
                "auto",
                "Sedan",
                "Coupe",
                "Sports Car",
                "Roadster",
                "Compact Car",
                "Hatchback",
                "Convertible",
                "Microcar",
                "City Car",
                "Smart Car",
                "motorcar",
                "Automobile",
            ],
            [
                "truck",
                "Lorry",
                "Tractor",
                "Trailer",
                "Semi",
                "Semi-trailer",
                "Semi-truck",
                "Truck Trailer",
                "Truck Tractor",
                "Semi-Trailer",
                "Hauler",
                "Tractor-Trailer",
                "Pickup Truck",
                "Delivery Truck",
                "Box Truck",
                "Dump Truck",
                "Flatbed Truck",
                "Garbage Truck",
            ],
            [
                "bus",
                "Coach",
                "Motorcoach",
                "Transit",
                "School Bus",
                "Shuttle",
                "Minibus",
                "Tour Bus",
                "Double-Decker",
                "City Bus",
                "Charter Bus",
            ],
            [
                "train",
                "Railroad",
                "Railway",
                "Locomotive",
                "Express",
                "Freight Train",
                "Subway",
                "Metro",
                "Commuter Train",
                "Tram",
                "Monorail",
            ],
            [
                "motorcycle",
                "Motorbike",
                "Chopper",
                "Scooter",
                "Moped",
                "Dirt Bike",
                "Motorcycle",
                "Roadster",
                "Motorized Bicycle",
            ],
            [
                "bicycle",
                "Bike",
                "Pushbike",
                "Pedal Bike",
                "Road Bike",
                "Mountain Bike",
                "BMX",
                "City Bike",
            ],
        ]

        self.suffixes = [
            ".",
            " in the city.",
            " from behind.",
            " from the side.",
            " from the front.",
            " at the evening.",
            " through a camera.",
            " on a screen.",
            " on a sunny day.",
            " on a bad day.",
            " on a billboard.",
            " on a poster.",
            " on a magazine.",
            " on a newspaper.",
            " on a computer screen.",
            " on a phone screen.",
            " on a tablet screen.",
            " on a TV screen.",
            " on a laptop screen.",
            " on a desktop screen.",
            " through a lens.",
            " through a window.",
            " in a crop.",
            " in a frame.",
            " in a picture.",
            " in a photo.",
            " in a painting.",
            " in a drawing.",
            " in a sketch.",
            " in a cartoon.",
            " in a comic.",
        ]

        self.prefixes = [
            "",
            "a bad photo of a ",
            "a photo of many ",
            "a sculpture of a ",
            "a photo of the hard to see ",
            "a low resolution photo of the ",
            "a rendering of a ",
            "graffiti of a ",
            "a bad photo of the ",
            "a cropped photo of the ",
            "a tattoo of a ",
            "the embroidered ",
            "a photo of a hard to see ",
            "a bright photo of a ",
            "a photo of a clean ",
            "a photo of a dirty ",
            "a dark photo of the ",
            "a drawing of a ",
            "a photo of my ",
            "the plastic ",
            "a photo of the cool ",
            "a close-up photo of a ",
            "a black and white photo of the ",
            "a painting of the ",
            "a painting of a ",
            "a pixelated photo of the ",
            "a sculpture of the ",
            "a bright photo of the ",
            "a cropped photo of a ",
            "a plastic ",
            "a photo of the dirty ",
            "a jpeg corrupted photo of a ",
            "a blurry photo of the ",
            "a photo of the ",
            "a good photo of the ",
            "a rendering of the ",
            "a photo of one ",
            "a doodle of a ",
            "a close-up photo of the ",
            "a photo of a ",
            "the origami ",
            "a sketch of a ",
            "a doodle of the ",
            "a origami ",
            "a low resolution photo of a ",
            "the toy ",
            "a rendition of the ",
            "a photo of the clean ",
            "a photo of a large ",
            "a rendition of a ",
            "a photo of a nice ",
            "a photo of a weird ",
            "a blurry photo of a ",
            "a cartoon ",
            "art of a ",
            "a sketch of the ",
            "a embroidered ",
            "a pixelated photo of a ",
            "itap of the ",
            "a jpeg corrupted photo of the ",
            "a good photo of a ",
            "a plushie ",
            "a photo of the nice ",
            "a photo of the small ",
            "a photo of the weird ",
            "the cartoon ",
            "art of the ",
            "a drawing of the ",
            "a photo of the large ",
            "a black and white photo of a ",
            "the plushie ",
            "a dark photo of a ",
            "itap of a ",
            "graffiti of the ",
            "a toy ",
            "itap of my ",
            "a photo of a cool ",
            "a photo of a small ",
            "a tattoo of the ",
        ]

        assert mask_classification, "Only support mask classification model"
        assert (
            num_queries == 20
        ), "Only support 20 queries, one for each class and one for void"

        self.mask_classification = mask_classification
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.clip_model, _ = clip.load("RN50", device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.initialize_query_embed()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def freeze_everything_except_output_FFN(self):
        # Freeze layers
        for p in self.parameters():
            p.requires_grad = False

        for p in self.class_embed.parameters():
            p.requires_grad = True
        for p in self.mask_embed.parameters():
            p.requires_grad = True

    def freeze_transformer_layers(self, layers_to_freeze):
        # Freeze layers
        for frozen_layer in layers_to_freeze:
            assert frozen_layer < self.num_layers, "Frozen layer index is out of range"
            for p in self.transformer_self_attention_layers[frozen_layer].parameters():
                p.requires_grad = False
            for p in self.transformer_cross_attention_layers[frozen_layer].parameters():
                p.requires_grad = False
            for p in self.transformer_ffn_layers[frozen_layer].parameters():
                p.requires_grad = False

    def generate_query_embed_weights(self):
        class_texts = []
        for i in range(len(self.cityscapes_classes)):
            random_class_idx = torch.randint(
                0, len(self.cityscapes_classes[i]), (1,)
            ).item()
            random_suffix_idx = torch.randint(0, len(self.suffixes), (1,)).item()
            random_prefix_idx = torch.randint(0, len(self.prefixes), (1,)).item()
            class_texts.append(
                (
                    self.prefixes[random_prefix_idx]
                    + self.cityscapes_classes[i][random_class_idx]
                    + self.suffixes[random_suffix_idx]
                ).lower()
            )

        tokens = clip.tokenize(class_texts).to("cuda")
        text_targets = self.clip_model.encode_text(tokens)
        text_targets = torch.cat([text_targets, torch.zeros(1, 1024).to("cuda")])

        # disable gradient for text targets
        for param in text_targets:
            param.requires_grad = False

        return text_targets

    def initialize_query_embed(self):
        cityscapes_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        composed_classnames = ["a photo of a " + c for c in cityscapes_classes]
        tokens = clip.tokenize(composed_classnames).to("cuda")
        text_targets = self.clip_model.encode_text(tokens)

        # append a zero vector for the background class
        text_targets = torch.cat([text_targets, torch.zeros(1, 1024).to("cuda")])

        # These are initialized with CLIP
        # num_querys goes to 19 / 20, for each class

        # Now they are initialized with the CLIP weights and then frozen
        self.query_feat = nn.Embedding(
            self.num_queries,
            self.hidden_dim,
            _weight=text_targets,
        )
        for p in self.query_feat.parameters():
            p.requires_grad = False

        # Made the query embedding unlearnable
        self.query_embed = None

    def initialize_query_embed_with_suffix(self, suffix="at night"):
        # use synonims
        cityscapes_classes = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        if isinstance(suffix, str):
            composed_classnames = [
                f"a photo of a {c} {suffix}" for c in cityscapes_classes
            ]
        else:
            assert len(suffix) == len(
                cityscapes_classes
            ), "Suffix must be a string or a list of strings with the same length as cityscapes_classes"
            composed_classnames = [
                f"a photo of a {c} {s}" for c, s in zip(cityscapes_classes, suffix)
            ]
        tokens = clip.tokenize(composed_classnames).to("cuda")
        text_targets = self.clip_model.encode_text(tokens)

        # append a zero vector for the background class
        text_targets = torch.cat([text_targets, torch.zeros(1, 1024).to("cuda")])

        # These are initialized with CLIP
        # num_querys goes to 19 / 20, for each class

        # Now they are initialized with the CLIP weights and then frozen
        self.query_feat = nn.Embedding(
            self.num_queries,
            self.hidden_dim,
            _weight=text_targets,
        )
        for p in self.query_feat.parameters():
            p.requires_grad = False

        # Made the query embedding unlearnable
        self.query_embed = None

    def initialize_query_embed_with_array(self, word_array):
        assert (
            len(word_array) == self.num_queries - 1
        ), "The array must have the same length as the number of queries"

        tokens = clip.tokenize(word_array).to("cuda")
        text_targets = self.clip_model.encode_text(tokens)

        # append a zero vector for the background class
        text_targets = torch.cat([text_targets, torch.zeros(1, 1024).to("cuda")])

        # These are initialized with CLIP
        # num_querys goes to 19 / 20, for each class

        # Now they are initialized with the CLIP weights and then frozen
        self.query_feat = nn.Embedding(
            self.num_queries,
            self.hidden_dim,
            _weight=text_targets,
        )
        for p in self.query_feat.parameters():
            p.requires_grad = False

        # Made the query embedding unlearnable
        self.query_embed = None

    def initialize_query_embed_syn(self):
        # use synonims
        cityscapes_classes = [
            "street",
            "pavement",
            "structure",
            "barrier",
            "railing",
            "post",
            "stoplight",
            "roadsign",
            "plants",
            "landscape",
            "heavens",
            "individual",
            "cyclist",
            "automobile",
            "lorry",
            "autobus",
            "locomotive",
            "motorbike",
            "bike",
        ]

        composed_classnames = ["a photo of a " + c for c in cityscapes_classes]
        tokens = clip.tokenize(composed_classnames).to("cuda")
        text_targets = self.clip_model.encode_text(tokens)

        # append a zero vector for the background class
        text_targets = torch.cat([text_targets, torch.zeros(1, 1024).to("cuda")])

        # These are initialized with CLIP
        # num_querys goes to 19 / 20, for each class

        # Now they are initialized with the CLIP weights and then frozen
        self.query_feat = nn.Embedding(
            self.num_queries,
            self.hidden_dim,
            _weight=text_targets,
        )
        for p in self.query_feat.parameters():
            p.requires_grad = False

        # Made the query embedding unlearnable
        self.query_embed = None

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        # This was Changed
        query_embed = self.query_embed  # .weight.unsqueeze(1).repeat(1, bs, 1)

        if not self.isInference:
            query_feat_weight = self.generate_query_embed_weights()
            output = query_feat_weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # # prediction heads on learnable query features
        # Dont add the Classes
        # Make the query features un learnable
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # Normalize the output of the decoder
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # Run a Linear Embedding to get the class logits
        outputs_class = self.class_embed(decoder_output)

        # Run a MLP to get the mask logits
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
