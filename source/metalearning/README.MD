Collecting demos:
python scripts/reinforcement_learning/rsl_rl/collect_demos.py     --task OmniFromDemo-UR5eRobotiq2f85-CollectDemos-v0     --num_envs 2     --checkpoint peg_state_rl_expert.pt     env.scene.insertive_object=peg     env.scene.receptive_object=peghole --headless --num_demos 4 --max_demos_before_saving 4

to debug, just add ./debug.sh in front and take out python

post process the data to cut off episodes that drag on too long:
python -m metalearning.tools.post_process_data /path/to/episodes_000000.pt
or
python -m metalearning.tools.post_process_data /path/to/episodes --out-dir /path/to/episodes_post

training:
scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniFromDemo-UR5eRobotiq2f85-Train-v0 \
    --num_envs 128 \
    --logger wandb \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole

supervised context training:
python scripts/reinforcement_learning/rsl_rl/train_supervised_context.py

distillation training:
scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniFromDemo-UR5eRobotiq2f85-Distillation-v0 \
    --num_envs 128 \
    --logger wandb \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole

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
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole

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

supervised eval:
python scripts/reinforcement_learning/rsl_rl/eval_demo_tracking.py --task OmniFromDemo-UR5eRobotiq2f85-SupervisedEval-v0 --num_rollouts 16 --num_envs 8 --supervised_context_checkpoint logs/rsl_rl/supervised_context/seed3_hd512_L8_H4/model_070000.pt --supervised_open_loop  --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole

visualizing context + inference rollouts:
python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
