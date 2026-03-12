from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.student_teacher import StudentTeacher
from rsl_rl.networks import HiddenState

from uwlab_rl.rsl_rl.long_context_ac import LongContextActorCritic
from uwlab_rl.rsl_rl.utils import normalize_cfg


class LongContextStudentTeacher(StudentTeacher):
    """Student-teacher policy with a LongContextActorCritic student and MLP teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_cfg: dict[str, Any] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        student_cfg = normalize_cfg(student_cfg)
        teacher_cfg = normalize_cfg(teacher_cfg)
        student_cfg.pop("class_name", None)
        teacher_cfg.pop("class_name", None)

        action_distribution = student_cfg.get("action_distribution", "normal")
        if action_distribution != "normal":
            raise ValueError("LongContextStudentTeacher supports regression only (action_distribution='normal').")

        if kwargs:
            print(
                "LongContextStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        student_hidden_dims = student_cfg.get("actor_hidden_dims", student_cfg.get("student_hidden_dims", [256, 256, 256]))
        teacher_hidden_dims = teacher_cfg.get("actor_hidden_dims", teacher_cfg.get("teacher_hidden_dims", [256, 256, 256]))
        activation = teacher_cfg.get("activation", student_cfg.get("activation", "elu"))
        init_noise_std = student_cfg.get("init_noise_std", 0.1)
        noise_std_type = student_cfg.get("noise_std_type", "scalar")
        student_obs_normalization = bool(
            student_cfg.get("actor_obs_normalization", student_cfg.get("student_obs_normalization", False))
        )
        teacher_obs_normalization = bool(
            teacher_cfg.get("actor_obs_normalization", teacher_cfg.get("teacher_obs_normalization", False))
        )

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            student_obs_normalization=student_obs_normalization,
            teacher_obs_normalization=teacher_obs_normalization,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        student_obs_groups = {
            "policy": obs_groups["policy"],
            "critic": obs_groups.get("critic", obs_groups["policy"]),
        }
        self.student = LongContextActorCritic(
            obs=obs,
            obs_groups=student_obs_groups,
            num_actions=num_actions,
            **student_cfg,
        )

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        if hasattr(self.student, "reset"):
            try:
                self.student.reset(dones)
            except TypeError:
                self.student.reset()

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = self.student.act(obs)
        if hasattr(self.student, "distribution"):
            self.distribution = self.student.distribution
        return actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actions = self.student.act_inference(obs)
        if hasattr(self.student, "distribution"):
            self.distribution = self.student.distribution
        return actions

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        if hasattr(self.teacher_obs_normalizer, "eval"):
            self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        self.student.update_normalization(obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load student/teacher params or map PPO actor params into the teacher."""
        if any(
            key.startswith("student.")
            or key.startswith("teacher.")
            or key.startswith("teacher_obs_normalizer.")
            for key in state_dict
        ):
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, "eval"):
                self.teacher_obs_normalizer.eval()
            return True

        if any("actor" in key for key in state_dict):
            teacher_state_dict: dict[str, Any] = {}
            teacher_obs_normalizer_state_dict: dict[str, Any] = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_state_dict[key.replace("actor.", "")] = value
                if key.startswith("actor_obs_normalizer."):
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            if teacher_obs_normalizer_state_dict and hasattr(self.teacher_obs_normalizer, "load_state_dict"):
                self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, "eval"):
                self.teacher_obs_normalizer.eval()
            return False

        raise ValueError("state_dict does not contain student, teacher, or PPO actor parameters.")
