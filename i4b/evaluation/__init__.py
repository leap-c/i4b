"""Closed-loop controller evaluation on I4B benchmark scenarios.

See ``docs/EVAL_SPEC.md`` for the observation contract and ``docs/DATA_SCHEMA.md`` for the
dataset tables.
"""

from .closed_loop_eval import run_evaluation
from .dataset import BenchmarkDataset, load_controller_data, load_dataset
from .scenario_env import FORECAST_CHANNELS, HISTORY_CHANNELS, STATE_CHANNELS, ScenarioEnv

__all__ = [
    "FORECAST_CHANNELS",
    "HISTORY_CHANNELS",
    "STATE_CHANNELS",
    "BenchmarkDataset",
    "ScenarioEnv",
    "load_controller_data",
    "load_dataset",
    "run_evaluation",
]
