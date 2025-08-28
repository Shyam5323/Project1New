"""Experiment phase package exposing public experiment classes for linting/tools.
Adding this resolves Pylint E0611 (no-name-in-module) by making 'experiments' a regular package.
"""
from .phase1_baseline import Phase1BaselineExperiment  # noqa: F401
from .phase2_noise import Phase2NoiseExperiment  # noqa: F401

__all__ = [
    'Phase1BaselineExperiment',
    'Phase2NoiseExperiment',
]
