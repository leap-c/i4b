"""Environment wrappers adapting i4b vectorized envs to RL library protocols."""

from i4b.rl.wrappers.base import I4bVecEnvWrapper
from i4b.rl.wrappers.rsl_rl import RslRlVecEnvWrapper

__all__ = ["I4bVecEnvWrapper", "RslRlVecEnvWrapper"]
