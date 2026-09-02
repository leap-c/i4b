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
| `src/i4b_bench/scenario_env.py` `ScenarioEnv.__init__` | Never passes `delta_t` to `RoomHeatEnv` — relies on the default |
| `src/i4b_bench/scenario_env.py` `ForecastProvider.get_forecast` (now `src/i4b_bench/forecast.py`) | Hardcodes `delta_t = 900` independently |
| `src/i4b_bench/dataset.py` `load_dataset` / `load_controller_data` | Pure I/O, no resolution check |
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


## The corpus' disturbances follow two superseded conventions

**Status:** known, recorded, not corrected in corpus v2

`exogenous.parquet` and every trajectory in the corpus were built before
`6075307`, which changed how `prepare_disturbances` derives the two
disturbance channels. The corpus therefore differs from what that function
produces today, in two independent ways:

| Channel | Corpus (v2) | `prepare_disturbances` today | Gap |
|---|---|---|---|
| `T_amb` | hourly reading **held** across the four 15-min steps | linearly **interpolated** | up to 6.0 K |
| internal gains | profile indexed by **naive UTC** | indexed by **local time** | up to 168.6 W |

The gains difference is a pure time shift of the whole internal-gains
profile — occupancy *and* appliances, since both are looked up by
`(hour, weekday/weekend)` — by each scenario's UTC offset, stepping at
each DST transition. Solar gains are unaffected: they follow true solar
position, which is tied to real UTC time.

**Consequence.** The corpus' simulated occupants run on a UTC clock rather
than their own: 2–3 h late in Sofia and Nicosia, 1–2 h in most of the
continent, 0–1 h in Dublin. Occupancy therefore does not line up with
daylight or with the solar gain it should coincide with.

**Risk:** low for scoring, real for transfer. `Qdot_gains` is a channel the
`perfect` view *hands* a model rather than one it must predict, so the room
responds to the gains the model sees and the learning problem stays
self-consistent. It is a realism defect, and would matter for sim-to-real.

**Enforced in code:** `scripts/generate_excitation_levels.py` reads
`exogenous.parquet` rather than re-deriving disturbances, so an appended
trajectory inherits the corpus' convention by construction. Any new
generator appending to an existing corpus must do the same — re-deriving
moves the scored channel by 0.187 K mean / 0.690 K max, the size of the
benchmark's MAE range.

**Clean-up path:** corpus v3, which means regenerating all 4,202
trajectories including the 2,674 MPC ones through the external collector.
`prepare_disturbances` already implements the corrected behaviour; nothing
in the library needs changing, only the data.
