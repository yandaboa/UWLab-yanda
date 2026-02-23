"""Tracking runner that separates rollout inference and PPO updates across two devices."""

from __future__ import annotations

import io
import os
import time
from collections import deque

import torch

from rsl_rl.utils import store_code_state

from uwlab_rl.rsl_rl.tracking_runner import TrackingOnPolicyRunner


class SplitDeviceTrackingOnPolicyRunner(TrackingOnPolicyRunner):
    """On-policy runner with inference on env device and PPO updates on runner device."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.rollout_policy = self._clone_policy_for_rollout()

    def _clone_policy_for_rollout(self):
        # deepcopy can fail on modules with non-leaf cached tensors; serialize/deserialize is more robust.
        buffer = io.BytesIO()
        torch.save(self.alg.policy, buffer)
        buffer.seek(0)
        try:
            rollout_policy = torch.load(buffer, map_location=self.env.device, weights_only=False)
        except TypeError:
            rollout_policy = torch.load(buffer, map_location=self.env.device)
        return rollout_policy

    def _sync_rollout_policy(self) -> None:
        self.rollout_policy.load_state_dict(self.alg.policy.state_dict())

    def _sync_rollout_normalizers(self) -> None:
        for attr_name in ("actor_obs_normalizer", "critic_obs_normalizer"):
            train_norm = getattr(self.alg.policy, attr_name, None)
            rollout_norm = getattr(self.rollout_policy, attr_name, None)
            assert train_norm is not None and rollout_norm is not None, "Normalizers must be present."
            assert hasattr(train_norm, "state_dict") and hasattr(rollout_norm, "load_state_dict"), "Normalizers must have state_dict and load_state_dict."
            rollout_norm.load_state_dict(train_norm.state_dict())

    def train_mode(self):
        super().train_mode()
        self.rollout_policy.train()

    def eval_mode(self):
        super().eval_mode()
        self.rollout_policy.eval()

    def _act_with_rollout_policy(self, obs_env) -> torch.Tensor:
        if self.rollout_policy.is_recurrent:
            self.alg.transition.hidden_states = self.rollout_policy.get_hidden_states()

        actions_env = self.rollout_policy.act(obs_env).detach()
        values = self.rollout_policy.evaluate(obs_env).detach()
        actions_log_prob = self.rollout_policy.get_actions_log_prob(actions_env).detach()

        self.alg.transition.actions = actions_env.to(self.device)
        self.alg.transition.values = values.to(self.device)
        self.alg.transition.actions_log_prob = actions_log_prob.to(self.device)
        if getattr(self.rollout_policy, "action_distribution", "normal") == "categorical":
            self.alg.transition.action_logits = self.rollout_policy.action_logits.detach().to(self.device)
            self.alg.transition.action_mean = None
            self.alg.transition.action_sigma = None
        else:
            self.alg.transition.action_mean = self.rollout_policy.action_mean.detach().to(self.device)
            self.alg.transition.action_sigma = self.rollout_policy.action_std.detach().to(self.device)
        return actions_env

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # noqa: C901
        self._prepare_logging_writer()
        self._setup_trajectory_logging()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_env = self.env.get_observations()
        obs = obs_env.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self._prev_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if self._trajectory_enabled and self._next_viz_iter is None:
            self._next_viz_iter = start_iter + int(self._trajectory_cfg.get("log_every_iters", 1))

        for it in range(start_iter, tot_iter):
            start = time.time()
            self._sync_rollout_policy()
            if self._trajectory_enabled and self._next_viz_iter is not None and it >= self._next_viz_iter:
                if self._trajectory_collector is not None and not self._trajectory_collector.is_listening:
                    self._trajectory_collector.arm()
                self._next_viz_iter = it + int(self._trajectory_cfg.get("log_every_iters", 1))
            logged_attention_entropy = False

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if self._trajectory_collector is not None and self._trajectory_collector.is_listening:
                        new_envs = torch.nonzero(self._prev_dones, as_tuple=True)[0]
                        self._trajectory_collector.start_new_episodes(new_envs)

                    obs_before_env = obs_env
                    actions_env = self._act_with_rollout_policy(obs_env)
                    self.alg.transition.observations = obs_env.to(self.device)
                    if self._should_log_attention_entropy(it, logged_attention_entropy):
                        logged_attention_entropy = self._log_attention_entropy(it)

                    obs_env, rewards_env, dones_env, extras = self.env.step(actions_env.to(self.env.device))
                    obs, rewards, dones = (
                        obs_env.to(self.device),
                        rewards_env.to(self.device),
                        dones_env.to(self.device),
                    )

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Keep rollout normalizers aligned with the train policy after train-policy normalization update.
                    self._sync_rollout_normalizers()

                    if self.rollout_policy.is_recurrent:
                        self.rollout_policy.reset(dones_env)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
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
                            actions_env.to(self.env.device),
                            rewards_env,
                            dones_env,
                        )
                        if batch and self._trajectory_logger is not None:
                            self._trajectory_logger.log_pairs(batch, step=it)

                    self._prev_dones = dones_env.to(torch.bool)

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
