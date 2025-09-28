#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

"""Chess-specific processor pipelines."""

from .chess_pick_place import build_chess_pick_place_pipeline

__all__ = [
    "build_chess_pick_place_pipeline",
]
