#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 
#SBATCH --mem=50G
python3 ../run_JAWA_expts.py --dataset wine --bias 0.53 --ntrial 10 --L2_lambda 8.0