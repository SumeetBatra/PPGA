#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c4
#SBATCH --output=tmp/method3_walker2d_%j.log
# the following params need to be set manually
# PROJECT_PATH: path to QDax root on your system
# ENV_NAME: which environment to run on (see qdax/environments/__init__.py for a list of all runnable envs)
# then from the project root dir you can run ./scripts/train_pga_me.sh

ENV_NAME="humanoid"
GRID_SIZE=50  # number of cells per archive dimension
SEED=1111


RUN_NAME="paper_qdppo_"$ENV_NAME"_seed_"$SEED
echo $RUN_NAME
srun python -m algorithm.train_qdppo --env_name=$ENV_NAME \
                                --rollout_length=128 \
                                --use_wandb=True \
                                --wandb_group=paper \
                                --num_dims=2 \
                                --seed=$SEED \
                                --anneal_lr=False \
                                --num_minibatches=8 \
                                --update_epochs=4 \
                                --normalize_obs=True \
                                --normalize_rewards=True \
                                --wandb_run_name=$RUN_NAME\
                                --popsize=300 \
                                --env_batch_size=3000 \
                                --learning_rate=0.0003 \
                                --vf_coef=2 \
                                --entropy_coef=0.0 \
                                --target_kl=0.008 \
                                --max_grad_norm=1 \
                                --total_iterations=2000 \
                                --dqd_algorithm=cma_maega \
                                --sigma0=0.5 \
                                --restart_rule=no_improvement \
                                --calc_gradient_iters=10 \
                                --move_mean_iters=10 \
                                --archive_lr=0.1 \
                                --threshold_min=200 \
                                --grid_size=$GRID_SIZE \
                                --logdir=./experiments/paper_qdppo_$ENV_NAME