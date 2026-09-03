"""The I4B benchmark: the corpus, the evaluation protocols, and the metrics.

Two evaluations over the same corpus, the same views and the same observation. **Open loop** asks
a model to predict, given a context and candidate control trajectories, and scores point accuracy
and control response. **Closed loop** asks a controller to act, and scores energy and comfort.

Entry points:

    eval_benchmark_open_loop(predictor, evaluation_set="benchmark-v1")
    eval_scenario_closed_loop / eval_benchmark_closed_loop

Open loop runs off a compiled **evaluation set** -- a definition, the cases the corpus and the
plant produced from it, and a manifest fingerprinting both -- so scoring a predictor needs
neither the corpus nor the simulator. Closed loop still drives the plant, and its settings live
in `config/closed_loop/`. Either way what must not vary between two runs is written down rather
than defaulted, so two people get comparable numbers without agreeing on anything first.

This package imports `i4b`; nothing in `i4b` imports it back.
"""

from .closed_loop_eval import (
    ClosedLoopBenchmark,
    closed_loop_setting,
    eval_benchmark_closed_loop,
    eval_scenario_closed_loop,
)
from .cases import case_schema, load_cases, load_manifest, validate_cases
from .control_gain import NOMINAL_ROLE, PLAN_ROLES, control_gain, gain_terms, probe_plans
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
from .evaluation_set import (
    OpenLoopDefinition,
    Window,
    load_definition,
    resolve_evaluation_set,
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
    Predictor,
    eval_benchmark_open_loop,
    inspect_case,
)
from .scenario_env import ScenarioEnv

__all__ = [
    "CONTROL_CHANNELS",
    "DISTURBANCE_CHANNELS",
    "PLANT_STATE_CHANNELS",
    "EXCITATION",
    "EXCITATION_DTYPE",
    "STATE_CHANNELS",
    "NOMINAL_ROLE",
    "PLAN_ROLES",
    "BenchmarkDataset",
    "ClosedLoopBenchmark",
    "OpenLoopDefinition",
    "ObsView",
    "Predictor",
    "Window",
    "ScenarioEnv",
    "aligned",
    "build_observation",
    "case_schema",
    "control_gain",
    "eval_benchmark_closed_loop",
    "eval_benchmark_open_loop",
    "eval_scenario_closed_loop",
    "inspect_case",
    "closed_loop_setting",
    "evaluation_scenarios",
    "gain_terms",
    "load_cases",
    "load_controller_data",
    "load_dataset",
    "load_definition",
    "load_manifest",
    "probe_plans",
    "read_window",
    "resolve_evaluation_set",
    "shard_index",
    "validate_cases",
]
