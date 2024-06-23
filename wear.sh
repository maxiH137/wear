#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

python main.py --config ./configs/60_frames_30_stride/tridet_inertial.yaml --seed 1 --eval_type split
