#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 
#SBATCH --mem=50G 
python3 ../run_JAWA_expts.py --dataset airfoil --bias 0.85 --ntrial 10 --L2_lambda 1.0