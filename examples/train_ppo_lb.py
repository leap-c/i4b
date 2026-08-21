"""GPU-native PPO with Log-Barrier constraint (safe RL) on i4b RoomHeat environments.

PPO-LB constrains temperature deviation (comfort cost) via a log-barrier
function, following the CSAC-LB design adapted to on-policy PPO.

Usage::

    # Default: 4096 envs, 200 iterations
    JAX_PLATFORMS=cuda python examples/train_ppo_lb.py

    # Custom configuration
    JAX_PLATFORMS=cuda python examples/train_ppo_lb.py \
        --num-envs 8192 \
        --iterations 500 \
        --cost-limit 0.02 \
        --save-dir runs/ppo_lb_8k
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from i4b.gym_interface.vec_env import RoomHeatVecEnv
from i4b.rl.wrappers.rsl_rl import RslRlVecEnvWrapper
from i4b.rl.algorithms.actor_critic import SafeActorCritic
from i4b.rl.algorithms.ppo_lb import PPOLB, PPOLBConfig
from i4b.rl.runners.on_policy_runner import OnPolicyRunner, RunnerConfig


def main():
    parser = argparse.ArgumentParser(
        description="GPU-native PPO with Log-Barrier constraint (safe RL) for room heating control"
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=32, help="Rollout length per env")
    parser.add_argument("--iterations", type=int, default=200, help="PPO update iterations")
    parser.add_argument("--building", type=str, default="sfh_2016_now_0_soc")
    parser.add_argument("--method", type=str, default="4R3C")
    parser.add_argument("--delta-t", type=int, default=900)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="runs/ppo_lb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0")

    # PPO-LB specific
    parser.add_argument("--cost-limit", type=float, default=0.05,
                        help="Maximum allowed mean comfort deviation (K) per step")
    parser.add_argument("--barrier-factor", type=float, default=3.0,
                        help="Log-barrier sharpness (larger = harder constraint)")
    parser.add_argument("--barrier-multiplier", type=float, default=1.0,
                        help="Scaling weight on barrier term in actor loss")
    parser.add_argument("--cost-gamma", type=float, default=0.99,
                        help="Discount factor for cost GAE")
    parser.add_argument("--cost-lam", type=float, default=0.95,
                        help="GAE lambda for cost returns")

    args = parser.parse_args()

    # --- Create JAX env ---
    jax_env = RoomHeatVecEnv(
        num_envs=args.num_envs,
        building=args.building,
        hp_model="Heatpump_AW",
        method=args.method,
        mdot_HP=0.25,
        internal_gain_profile="i4b_data/profiles/InternalGains/ResidentialDetached.csv",
        delta_t=args.delta_t,
        days=args.days,
        device="gpu",
        performance_mode=True,
    )
    print(f"Created JAX env: {args.num_envs} envs, obs_dim={jax_env.single_observation_space.shape[0]}")

    # --- Wrap for PyTorch ---
    env = RslRlVecEnvWrapper(jax_env, torch_device=args.device)

    # --- Build safe actor-critic + PPO-LB ---
    actor_critic = SafeActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        cost_critic_hidden_dims=(256, 256, 256),
    )
    ppo_cfg = PPOLBConfig(
        learning_rate=args.lr,
        cost_limit=args.cost_limit,
        barrier_factor=args.barrier_factor,
        barrier_multiplier=args.barrier_multiplier,
        cost_gamma=args.cost_gamma,
        cost_lam=args.cost_lam,
    )
    ppo = PPOLB(actor_critic, cfg=ppo_cfg, device=args.device)

    # --- Train ---
    runner_cfg = RunnerConfig(
        num_steps_per_env=args.num_steps,
        save_dir=args.save_dir,
        device=args.device,
        log_interval=1,
        save_interval=50,
    )
    runner = OnPolicyRunner(env, ppo, cfg=runner_cfg)

    total_steps = args.iterations * args.num_steps * args.num_envs
    print(f"Starting PPO-LB training: {args.iterations} iterations, "
          f"{total_steps:,} total env steps")
    print(f"Device: {args.device}")
    print(f"Cost limit (max comfort deviation): {args.cost_limit} K/step")
    print(f"Barrier factor: {args.barrier_factor}")

    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
