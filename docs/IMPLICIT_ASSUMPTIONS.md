# Implicit assumptions

Tracked assumptions that are not yet enforced in code. Each entry should
link to the files involved so a future refactor can find them quickly.

## Temporal resolution is always 900 s (15 min)

**Status:** unvalidated end-to-end

The evaluation pipeline assumes a 15-minute timestep in several independent
places without a single source of truth:

| Location | How it enters |
|---|---|
| `i4b/gym_interface/room_env.py` (`delta_t=900` default) | Constructor default; validated against disturbance DataFrame spacing |
| `i4b/evaluation/scenario_env.py` `ScenarioEnv.__init__` | Never passes `delta_t` to `RoomHeatEnv` — relies on the default |
| `i4b/evaluation/scenario_env.py` `ForecastProvider.get_forecast` | Hardcodes `delta_t = 900` independently |
| `i4b/evaluation/dataset.py` `load_dataset` / `load_controller_data` | Pure I/O, no resolution check |
| Controller trajectory (history seeding) | Assumed to match but never validated |

`RoomHeatEnv` is the only component that validates spacing (it raises if the
disturbances DataFrame doesn't match `self.delta_t`). The forecast provider
and history seeding path have no such guard.

**Risk:** low while we only use the current benchmark (which is 15-min
throughout). Would become a silent-bug source if datasets with different
resolutions are introduced.

**Clean-up path:** introduce a single `delta_t` that flows from the dataset
metadata through `ScenarioEnv` to both `RoomHeatEnv` and
`ForecastProvider`, and validate the controller trajectory spacing on load.
