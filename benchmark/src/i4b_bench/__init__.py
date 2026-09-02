"""The I4B benchmark: the corpus, the evaluation protocols, and the metrics.

Two evaluations over the same corpus, the same views and the same observation. **Open loop** asks
a model to predict, given a context and candidate control trajectories, and scores point accuracy
and control response. **Closed loop** asks a controller to act, and scores energy and comfort.

Each has a scenario-level and a benchmark-level entry point:

    eval_scenario_open_loop / eval_benchmark_open_loop
    eval_scenario_closed_loop / eval_benchmark_closed_loop

`BENCHMARK` fixes what must not vary between runs, so two people calling a benchmark-level
function get comparable numbers without agreeing on anything first.

This package imports `i4b`; nothing in `i4b` imports it back.
"""

from .closed_loop_eval import (
    ClosedLoopBenchmark,
    closed_loop_setting,
    eval_benchmark_closed_loop,
    eval_scenario_closed_loop,
)
from .control_gain import control_gain, gain_terms, probe_plans
from .dataset import (
    EXCITATION,
    EXCITATION_DTYPE,
    BenchmarkDataset,
    aligned,
    evaluation_scenarios,
    load_controller_data,
    load_dataset,
    read_window,
    shard_index,
)
from .observation import (
    CONTROL_CHANNELS,
    DISTURBANCE_CHANNELS,
    PLANT_STATE_CHANNELS,
    STATE_CHANNELS,
    ObsView,
    build_observation,
)
from .open_loop_eval import (
    OpenLoopBenchmark,
    Predictor,
    Window,
    eval_benchmark_open_loop,
    eval_scenario_open_loop,
    open_loop_setting,
)
from .scenario_env import ScenarioEnv

__all__ = [
    "CONTROL_CHANNELS",
    "DISTURBANCE_CHANNELS",
    "PLANT_STATE_CHANNELS",
    "EXCITATION",
    "EXCITATION_DTYPE",
    "STATE_CHANNELS",
    "BenchmarkDataset",
    "ClosedLoopBenchmark",
    "OpenLoopBenchmark",
    "ObsView",
    "Predictor",
    "Window",
    "ScenarioEnv",
    "aligned",
    "build_observation",
    "control_gain",
    "eval_benchmark_closed_loop",
    "eval_benchmark_open_loop",
    "eval_scenario_open_loop",
    "eval_scenario_closed_loop",
    "open_loop_setting",
    "closed_loop_setting",
    "evaluation_scenarios",
    "gain_terms",
    "load_controller_data",
    "load_dataset",
    "probe_plans",
    "read_window",
    "shard_index",
]
