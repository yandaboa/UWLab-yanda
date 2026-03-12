# Raw State Capture (Concise)

## Where it lives
- `metalearning.state_storage.StateStorage` handles raw state capture.
- `collect_demos.py` forwards CLI flags and attaches `raw_states` to rollouts.

## Flags
- `--save_raw_states` (default: True)
- `--state_capture_interval` (default: 1)
- `context.use_raw_states` (default: False)

## Stored format
In each episode file:
- `episode["raw_states"]` is a list of entries: `{"timestep": int, "state": dict}`.
- `state` matches `env.scene.get_state(is_relative=True)` for a single env:
  - `articulation.<asset>.root_pose`, `root_velocity`, `joint_position`, `joint_velocity`
  - `rigid_object.<asset>.root_pose`, `root_velocity`

## Reset usage (later)
1. Move tensors to the env device.
2. Call `env.scene.reset_to(state, env_ids=..., is_relative=True)`.
3. Set `env.episode_length_buf[env_ids]` to the saved `timestep`.
