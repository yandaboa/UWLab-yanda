# Metalearning Quick Commands

## 1) Collect demos

```bash
python scripts/reinforcement_learning/rsl_rl/collect_demos.py \
  --task OmniFromDemo-UR5eRobotiq2f85-CollectDemos-v0 \
  --num_envs 2 \
  --checkpoint peg_state_rl_expert.pt \
  --headless \
  --num_demos 4 \
  --max_demos_before_saving 4 \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

Debug tip: prepend `./debug.sh` and remove `python`.

## 2) Post-process episodes

Cut off episodes that drag on too long:

```bash
python -m metalearning.tools.post_process_data /path/to/episodes_000000.pt
```

Or process a directory:

```bash
python -m metalearning.tools.post_process_data /path/to/episodes --out-dir /path/to/episodes_post
```

## 3) Training

### RL training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniFromDemo-UR5eRobotiq2f85-Train-v0 \
  --num_envs 128 \
  --logger wandb \
  --headless \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

### Supervised context training

```bash
python scripts/reinforcement_learning/rsl_rl/train_supervised_context.py
```

### Forward dynamics (residual) training

Uses `scripts/reinforcement_learning/rsl_rl/train_forward_dynamics.py` to learn:
`||(s_t+1 - s_t) - f(s_t, a_t)||^2`.

```bash
python scripts/reinforcement_learning/rsl_rl/train_forward_dynamics.py \
  --seed 0 \
  data.train_episode_paths='["episodes/20260222_155251/episodes_000001_trim.pt","episodes/20260222_155251/episodes_000002_trim.pt"]' \
  data.validation_episode_paths='["episodes/20260222_155251/episodes_000000_trim.pt"]' \
  data.batch_size=256 \
  model.num_layers=4 \
  model.hidden_dim=512 \
  optim.num_steps=150000 \
  optim.learning_rate=3e-4 \
  optim.weight_decay=1e-4 \
  logging.experiment_name=forward_dynamics \
  logging.run_name=fd_run_01 \
  logging.use_wandb=true
```

### Distillation training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniFromDemo-UR5eRobotiq2f85-Distillation-v0 \
  --num_envs 128 \
  --logger wandb \
  --headless \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

### Distributed training

```bash
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node 2 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniFromDemo-UR5eRobotiq2f85-Train-v0 \
  --num_envs 256 \
  --logger wandb \
  --headless \
  --distributed \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

If the port is taken, add `--master_port 29501` before `scripts/reinforcement_learning/rsl_rl/train.py`.

### Train experts

```bash
python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node 4 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --distributed \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

## 4) Evaluation

### Demo-tracking eval

```bash
python scripts/reinforcement_learning/rsl_rl/eval_demo_tracking.py \
  --task OmniFromDemo-UR5eRobotiq2f85-Eval-v0 \
  --num_rollouts 50 \
  --num_envs 32 \
  --checkpoint cube_tracking_transformer_actor.pt \
  --headless \
  env.scene.insertive_object=rectangle \
  env.scene.receptive_object=wall
```

### Supervised eval

```bash
python scripts/reinforcement_learning/rsl_rl/eval_demo_tracking.py \
  --task OmniFromDemo-UR5eRobotiq2f85-SupervisedEval-v0 \
  --num_rollouts 16 \
  --num_envs 8 \
  --supervised_context_checkpoint logs/rsl_rl/supervised_context/seed3_hd512_L8_H4/model_070000.pt \
  --supervised_open_loop \
  --headless \
  env.scene.insertive_object=peg \
  env.scene.receptive_object=peghole
```

## 5) Visualize trajectories

```bash
python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
```
