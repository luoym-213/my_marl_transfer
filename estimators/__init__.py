"""Communication-aware teammate state estimators for Chapter 3."""

from estimators.base import EstimateBatch, StateEstimator, StaleEstimator
from estimators.kalman import ConstantVelocityKalmanEstimator

try:
    from estimators.commdrop import CommDropEstimator, CommDropModel
except ImportError:  # Keep the NumPy-only KF usable without PyTorch.
    CommDropEstimator = None
    CommDropModel = None

__all__ = [
    "EstimateBatch",
    "StateEstimator",
    "StaleEstimator",
    "ConstantVelocityKalmanEstimator",
    "CommDropEstimator",
    "CommDropModel",
]
