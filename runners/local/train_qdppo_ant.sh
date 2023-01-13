#!/bin/sh

# the following params need to be set manually
# PROJECT_PATH: path to QDax root on your system
# ENV_NAME: which environment to run on (see qdax/environments/__init__.py for a list of all runnable envs)
# then from the project root dir you can run ./scripts/train_pga_me.sh

ENV_NAME="ant"
GRID_SIZE=10  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_qdppo_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
python -m algorithm.train_qdppo --env_name=$ENV_NAME \
                                --rollout_length=128 \
                                --use_wandb=True \
                                --seed=$SEED \
                                --wandb_group=paper \
                                --num_dims=4 \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=False \
                                --normalize_rewards=True \
                                --wandb_run_name=$RUN_NAME \
                                --popsize=400 \
                                --env_batch_size=4000 \
                                --learning_rate=0.001 \
                                --vf_coef=2 \
                                --max_grad_norm=1 \
                                --torch_deterministic=False \
                                --total_iterations=2000 \
                                --dqd_algorithm=cma_maega \
                                --calc_gradient_iters=10 \
                                --move_mean_iters=10 \
                                --archive_lr=1.0 \
                                --restart_rule=no_improvement \
                                --sigma0=3.0 \
                                --threshold_min=-10000 \
                                --grid_size=$GRID_SIZE \
                                --logdir=./experiments/paper_qdppo_$ENV_NAME