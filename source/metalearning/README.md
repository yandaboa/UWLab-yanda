# Metalearning: OmniFromDemo UR5e Robotiq

This document explains what happens when running `scripts/reinforcement_learning/rsl_rl/train.py` with
`--task OmniFromDemo-UR5eRobotiq2f85-Train-v0`, and what each reward/observation/context term in
`source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/from_demo/config/ur5e_robotiq_2f85/rl_state_cfg.py`
does.

## Training flow (high level)

1. `train.py` parses CLI, launches Isaac Sim, and loads task + agent configs via the hydra entry points.
2. The task ID `OmniFromDemo-UR5eRobotiq2f85-Train-v0` is registered to `FromDemoEnv` with
   `Ur5eRobotiq2f85RelJointPosFromDemoTrainCfg` and the agent config `PPOWithContextRunnerCfg`.
3. The environment is created via `gym.make(task_id, cfg=env_cfg)`, then wrapped for RSL-RL.
4. The PPO runner is built with `LongContextActorCritic` and trained for `max_iterations`.

The training objective is to track a recorded demo trajectory by matching robot link orientations and end-effector
positions over time.

## Demo context (what the model is tracking)

The training config provides `DemoContextCfg` with a list of episode files. These files store recorded episodes that
include `obs`, `actions`, `rewards`, `states`, and `physics`. `FromDemoEnv` constructs a `DemoTrackingContext`, which:

- Loads all episodes from the specified `.pt` files.
- Infers obs/action/reward shapes and pads sequences to a shared max length.
- On reset, samples an episode per environment and:
  - Populates demo buffers (`demo_obs`, `demo_actions`, `demo_rewards`).
  - Resets the scene to the demo `states` (optionally with noise).
  - Reapplies the recorded physics parameters.

The `resample_episode` event is triggered on reset so each environment gets a new demo.

## Observations (what the policy and critic see)

### Base observation terms (policy + critic)

These are defined in `ObservationsCfg` and used by multiple task variants:

- `prev_actions`: last action applied to the environment (`env.action_manager.action`).
- `joint_pos`: robot joint positions for the configured joints.
- `end_effector_pose`: pose of the robot end-effector in the robot root frame. This uses metadata offsets for the
  robot root and gripper, and returns axis-angle rotation.
- `insertive_asset_pose`: pose of the insertive object relative to the gripper frame (with metadata offsets).
- `receptive_asset_pose`: pose of the receptive object relative to the gripper frame (axis-angle).
- `insertive_asset_in_receptive_asset_frame`: pose of the insertive object in the receptive object's frame.

### Critic-only terms

`CriticCfg` adds privileged state and physics parameters:

- `time_left`: 1 - (elapsed_steps / max_episode_length).
- `joint_vel`: robot joint velocities.
- `end_effector_vel_lin_ang_b`: end-effector linear + angular velocity in the robot root frame.
- `robot_material_properties`, `insertive_object_material_properties`, `receptive_object_material_properties`,
  `table_material_properties`: PhysX material properties (flattened).
- `robot_mass`, `insertive_object_mass`, `receptive_object_mass`, `table_mass`: masses.
- `robot_joint_friction`, `robot_joint_armature`, `robot_joint_stiffness`, `robot_joint_damping`: joint/actuator params.

### Training observations (used by OmniFromDemo-Train)

`TrainingObservationsCfg` extends the debug group and adds three extra groups:

- `critic` (TrackingCriticCfg):
  - `context_obs`: demo observation at `t + 1` (supervision target).
  - `context_actions`: demo action at `t`.
  - `context_rewards`: demo reward at `t + 1`.
- `context` (ContextCfg):
  - `context_obs`: full padded demo observation sequence.
  - `context_actions`: full padded demo action sequence.
  - `context_rewards`: full padded demo reward sequence.
  - `context_lengths`: demo sequence lengths.
- `ee_pose` (EndEffectorPoseCfg):
  - `end_effector_pose`: same as the base end-effector pose term, used for tracking reward.

### Debug observations

`DebugObservationsCfg` exposes a small, non-concatenated view of end-effector, insertive, and receptive poses for
inspection/logging.

## Rewards (what is optimized)

### Base task rewards (standard reset-state tasks)

`RewardsCfg` defines generic manipulation rewards:

- `action_magnitude`: L2 penalty on the current action vector (clamped).
- `action_rate`: L2 penalty on the action delta between steps (clamped).
- `joint_vel`: L2 penalty on arm joint velocities (clamped).
- `abnormal_robot`: heavy penalty if joint velocities exceed limits.
- `progress_context`: computes alignment between insertive and receptive objects using their assembled offsets and
  marks success based on position + orientation thresholds.
- `ee_asset_distance`: tanh-shaped distance from end-effector to insertive object using metadata offsets.
- `dense_success_reward`: exp(-distance/std) for position and orientation error (averaged).
- `success_reward`: 1 when the success condition first becomes true.

### Demo-tracking rewards (used in OmniFromDemo-Train)

`FromDemoRewardsCfg` replaces base rewards with demo tracking terms:

- `tracking_joint_angle` (`pose_quat_tracking_reward`):
  - Uses demo link quaternions at the current timestep.
  - Computes relative rotation between demo and current link quats.
  - Returns `exp(-k * sum(angle^2))` across links.
- `tracking_end_effector`:
  - Uses demo end-effector position at the current timestep.
  - Returns `exp(-k * ||p_demo - p_curr||^2)`.

## Commands and metrics

`DemoTrackingMetricsCommand` logs:

- `joint_tracking_error`: sum of squared per-link angle errors.
- `ee_tracking_error`: squared distance between demo and current end-effector position.

This command is added to `CommandsCfg` for logging/diagnostics.

## Terminations

For the demo-tracking task:

- `time_out`: episode length exceeded.
- `abnormal_robot`: joint velocities exceed limits.

Demo collection adds `terminate_on_success` with a delay to end episodes after success.

## Events and reset behavior

Several event configs are defined in the same file:

- `BaseEventCfg`: randomizes friction, restitution, mass, and joint/actuator parameters at startup/reset.
- `TrainEventCfg`: resets from multi-source datasets using `MultiResetManager`.
- `FromDemoTrainEventCfg`: resamples demo episodes on reset.
- `FromDemoEvalEventCfg`: resamples demo episodes on reset for evaluation.
- `DemoCollectEventCfg`: resamples environment noise after reset.

The demo-tracking training variant uses `FromDemoTrainEventCfg`, so it focuses on demo resampling rather than physics
randomization.
