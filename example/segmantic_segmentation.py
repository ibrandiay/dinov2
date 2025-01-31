# Copyright (c) Meta Platforms, Inc. and affiliates
"""
	Semantic Segmentation Example for Dinov2
"""

import torch
import sys
import math
import itertools
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import init_segmentor, inference_segmentor
from functools import partial
import numpy as np
from typing import Sequence, Tuple, Union
from PIL import Image
import dinov2.eval.segmentation.models
from dinov2.models.vision_transformer import vit_small


def create_segmenter(cfg, backbone_model):
	model = init_segmentor(cfg)
	model.backbone.forward = partial(
		backbone_model.get_intermediate_layers,
		n=cfg.model.backbone.out_indices,
		reshape=True,
	)
	if hasattr(backbone_model, "patch_size"):
		model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
	model.init_weights()
	return model


def load_model(model_path: str) -> torch.nn.Module:
	"""
	Load all model weights: backbone and head
	Args:
		model_path (str): path to model
	Returns:
		Tuple[torch.nn.Module, torch.nn.Module]: backbone and head
	"""
	
	# Load backbone
	backbone = vit_small()
	backbone.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
	backbone.eval().cuda()
	
	# Load head
	head = dinov2.eval.segmentation.models.build_head()
	head.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
	head.eval().cuda()
	
	return backbone, head


def main(model_path: str, image_path: str):
	model = load_model(model_path)
	image = Image.open(image_path)
	segmentation = dinov2.eval.segmentation.models.predict(model, image)