"""Helpers for the Car Price project: feature engineering and the outlier clipper.

The saved model pickles references to these functions/classes, so the package must be
importable wherever the model is loaded (the notebook folder's parent, or the app folder).
"""
from .features import REF_YEAR, add_features
from .outlier_handler import OutlierClipper

__all__ = ["REF_YEAR", "add_features", "OutlierClipper"]
