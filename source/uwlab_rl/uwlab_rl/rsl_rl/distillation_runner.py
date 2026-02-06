from __future__ import annotations

from typing import Any

from tensordict import TensorDict

from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners.distillation_runner import DistillationRunner as BaseDistillationRunner

from uwlab_rl.rsl_rl.distillation import Distillation
from uwlab_rl.rsl_rl.student_teacher import LongContextStudentTeacher
from uwlab_rl.rsl_rl.utils import normalize_cfg


class DistillationRunner(BaseDistillationRunner):
    """Repo-local distillation runner with long-context student support."""

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        policy_cfg = dict(self.policy_cfg)
        alg_cfg = dict(self.alg_cfg)

        policy_class_name = policy_cfg.pop("class_name")
        student_cfg = normalize_cfg(policy_cfg.pop("student_cfg", None))
        teacher_cfg = normalize_cfg(policy_cfg.pop("teacher_cfg", None))
        policy_cfg.pop("action_discretization_spec", None)

        policy_class_map: dict[str, type] = {
            "LongContextStudentTeacher": LongContextStudentTeacher,
            "StudentTeacher": StudentTeacher,
            "StudentTeacherRecurrent": StudentTeacherRecurrent,
        }
        student_teacher_class = policy_class_map.get(policy_class_name)
        if student_teacher_class is None:
            student_teacher_class = eval(policy_class_name)

        init_kwargs = dict(policy_cfg)
        if student_cfg:
            init_kwargs["student_cfg"] = student_cfg
        if teacher_cfg:
            init_kwargs["teacher_cfg"] = teacher_cfg

        student_teacher = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **init_kwargs
        ).to(self.device)

        alg_class_name = alg_cfg.pop("class_name")
        alg_class_map: dict[str, type] = {"Distillation": Distillation}
        alg_class = alg_class_map.get(alg_class_name)
        if alg_class is None:
            alg_class = eval(alg_class_name)

        if alg_class is Distillation:
            alg_cfg.setdefault("transformer_optimizer_cfg", getattr(student_teacher, "transformer_optimizer_cfg", None))
            alg_cfg.setdefault("num_learning_iterations", self.cfg.get("max_iterations"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            (self.env.num_actions,),
        )

        return alg
