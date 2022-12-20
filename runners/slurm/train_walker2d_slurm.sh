#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c4
#SBATCH --output=tmp/method3_walker2d_%j.log

srun python -m algorithm.train_qdppo \
            --env_name=walker2d \
            --algorithm=qd-ppo \
            --seed=0000 \
            --rollout_length=128 \
            --use_wandb=True \
            --wandb_group=QDPPO \
            --num_dims=2 \
            --num_minibatches=4 \
            --update_epochs=2 \
            --normalize_obs=False \
            --normalize_rewards=True \
            --wandb_run_name=qdppo_walker2d_12_13_22 \
            --mega_lambda=200 \
            --env_type=brax \
            --env_batch_size=3000 \
            --learning_rate=0.001 \
            --vf_coef=2 \
            --max_grad_norm=1 \
            --dqd_lr=0.01 \
            --pretrain=False \
            --total_iterations=1000 \
            --dqd_algorithm=cma_maega \
            --sigma0=1.0 \
            --restart_rule=no_improvement
            --torch_deterministic=False \
            --logdir=./logs/qdppo_walker2d_12_13_22 \
