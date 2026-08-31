"""Closed-loop controller evaluation on I4B benchmark scenarios.

See ``docs/EVAL_SPEC.md`` for the observation contract and ``docs/DATA_SCHEMA.md`` for the
dataset tables.
"""

from .closed_loop_eval import run_evaluation
from .dataset import (
    BenchmarkDataset,
    evaluation_scenarios,
    load_controller_data,
    load_dataset,
)
from .scenario_env import CONTROL_CHANNELS, STATE_CHANNELS, ObsView, ScenarioEnv

__all__ = [
    "CONTROL_CHANNELS",
    "STATE_CHANNELS",
    "ObsView",
    "BenchmarkDataset",
    "ScenarioEnv",
    "evaluation_scenarios",
    "load_controller_data",
    "load_dataset",
    "run_evaluation",
]
