"""Custom RSL-RL runner with trajectory visualization hooks."""

from __future__ import annotations

import os
import time
from collections import deque

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from metalearning.logger import WandbTrajectoryLogger
from metalearning.trajectory_logging import TrajectoryPairCollector


class TrajectoryOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that logs demo-vs-rollout trajectories."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._trajectory_cfg = train_cfg.get("trajectory_viz") or {}
        self._trajectory_enabled = bool(self._trajectory_cfg.get("enable", False))
        if self.disable_logs:
            self._trajectory_enabled = False
        self._trajectory_collector: TrajectoryPairCollector | None = None
        self._trajectory_logger: WandbTrajectoryLogger | None = None
        self._next_viz_iter: int | None = None
        self._prev_dones: torch.Tensor | None = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        self._setup_trajectory_logging()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs_env = self.env.get_observations()
        obs = obs_env.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self._prev_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if self._trajectory_enabled and self._next_viz_iter is None:
            self._next_viz_iter = start_iter + int(self._trajectory_cfg.get("log_every_iters", 1))
        for it in range(start_iter, tot_iter):
            start = time.time()
            if self._trajectory_enabled and self._next_viz_iter is not None and it >= self._next_viz_iter:
                if self._trajectory_collector is not None and not self._trajectory_collector.is_listening:
                    self._trajectory_collector.arm()
                self._next_viz_iter = it + int(self._trajectory_cfg.get("log_every_iters", 1))

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if self._trajectory_collector is not None and self._trajectory_collector.is_listening:
                        new_envs = torch.nonzero(self._prev_dones, as_tuple=True)[0]
                        self._trajectory_collector.start_new_episodes(new_envs)

                    obs_before_env = obs_env
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs_env, rewards_env, dones_env, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (
                        obs_env.to(self.device),
                        rewards_env.to(self.device),
                        dones_env.to(self.device),
                    )
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                    if self._trajectory_collector is not None and self._trajectory_collector.is_listening:
                        batch = self._trajectory_collector.record_step(
                            obs_before_env,
                            actions.to(self.env.device),
                            rewards_env,
                            dones_env,
                        )
                        if batch and self._trajectory_logger is not None:
                            self._trajectory_logger.log_pairs(batch, step=it)

                    self._prev_dones = dones_env.to(torch.bool)

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _setup_trajectory_logging(self) -> None:
        if not self._trajectory_enabled:
            return
        obs_key = self._trajectory_cfg.get("obs_key", "debug/end_effector_pose")
        demo_obs_key = self._trajectory_cfg.get("demo_obs_key", "end_effector_pose")
        max_pairs = int(self._trajectory_cfg.get("max_pairs_per_log", 1))
        self._trajectory_collector = TrajectoryPairCollector(
            env=self.env,
            obs_key=obs_key,
            max_pairs_per_log=max_pairs,
        )
        self._trajectory_logger = WandbTrajectoryLogger(
            enable=self.logger_type == "wandb",
            agent_cfg=self.cfg,
            log_dir=self.log_dir or "",
            task_name=str(getattr(self.env.cfg, "name", "")),
            obs_key=obs_key,
            demo_obs_key=demo_obs_key,
            save_pairs=bool(self._trajectory_cfg.get("save_pairs", False)),
            max_pairs_per_log=max_pairs,
        )
