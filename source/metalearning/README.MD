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