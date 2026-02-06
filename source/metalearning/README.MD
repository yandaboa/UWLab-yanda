Collecting demos:
python scripts/reinforcement_learning/rsl_rl/collect_demos.py     --task OmniFromDemo-UR5eRobotiq2f85-CollectDemos-v0     --num_envs 2     --checkpoint rectangle_state_rl_expert.pt     env.scene.insertive_object=rectangle     env.scene.receptive_object=wall --headless

to debug, just add ./debug.sh in front and take out python

training:
scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniFromDemo-UR5eRobotiq2f85-Train-v0 \
    --num_envs 128 \
    --logger wandb \
    --headless \
    env.scene.insertive_object=rectangle \
    env.scene.receptive_object=wall

distributed training:
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniFromDemo-UR5eRobotiq2f85-Train-v0 \
    --num_envs 256 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=rectangle \
    env.scene.receptive_object=wall

if training won't start because the port is already taken, pass in --master_port 29501 before  scripts/reinforcement_learning/rsl_rl/train.py

training experts:
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=rectangle \
    env.scene.receptive_object=walldd

running evals to see how well something tracks a demo:
python scripts/reinforcement_learning/rsl_rl/eval_demo_tracking.py --task OmniFromDemo-UR5eRobotiq2f85-Eval-v0 --num_rollouts 50 --num_envs 32 --checkpoint cube_tracking_transformer_actor.pt --headless env.scene.insertive_object=rectangle env.scene.receptive_object=wall

visualizing context + inference rollouts:
python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
