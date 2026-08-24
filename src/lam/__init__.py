"""Standalone AdaWorld-style LAM pretraining entrypoints.

The actual feature-space LAM module remains in ``lvr.modules.lam`` so that
the pretraining and downstream LVR paths share exactly the same weights.
"""

from .model import FeatureLAM

__all__ = ["FeatureLAM"]
