"""
Multi-PTM binary classifiers on top of shared PTM-Mamba hidden states.

Train one model on pooled data from several PTM-specific feature sets,
optionally conditioning on PTM id, and report metrics per PTM.
"""

from multip_ptm_classification.ptm_token_map import (
    DEFAULT_PTM_TOKEN_MAP,
    get_ptm_token_for_folder,
)
from multip_ptm_classification.datasets import MultiPTMFeaturesDataset

__all__ = [
    "DEFAULT_PTM_TOKEN_MAP",
    "get_ptm_token_for_folder",
    "MultiPTMFeaturesDataset",
]
