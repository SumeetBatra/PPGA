#!/bin/sh

# the following params need to be set manually
# PROJECT_PATH: path to QDax root on your system
# ENV_NAME: which environment to run on (see qdax/environments/__init__.py for a list of all runnable envs)
# then from the project root dir you can run ./scripts/train_pga_me.sh

ENV_NAME="walker2d"
GRID_SIZE=50  # number of cells per archive dimension

set -- 1111 2222 3333 4444  # seeds

for item in "$@";
 do echo "Running seed $item";
 RUN_NAME="paper_qdppo_"$ENV_NAME"_seed_"$item
 echo $RUN_NAME
 python -m algorithm.train_qdppo --env_name=walker2d \
                                 --rollout_length=128 \
                                 --use_wandb=True \
                                 --wandb_group=paper \
                                 --seed="$item" \
                                 --num_dims=2 \
                                 --num_minibatches=8 \
                                 --update_epochs=4 \
                                 --normalize_obs=False \
                                 --normalize_rewards=True \
                                 --wandb_run_name=$RUN_NAME \
                                 --popsize=300 \
                                 --grid_size=50 \
                                 --env_batch_size=3000 \
                                 --learning_rate=0.001 \
                                 --vf_coef=2 \
                                 --max_grad_norm=1 \
                                 --total_iterations=2000 \
                                 --dqd_algorithm=cma_maega \
                                 --torch_deterministic=False \
                                 --archive_lr=0.15 \
                                 --threshold_min=0.0 \
                                 --restart_rule=no_improvement \
                                 --calc_gradient_iters=10 \
                                 --move_mean_iters=10 \
                                 --logdir=./experiments/paper_qdppo_walker2d
done