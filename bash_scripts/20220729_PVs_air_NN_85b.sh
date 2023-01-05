#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 
#SBATCH --mem=50G 
python3 ../run_JAW_PredictiveValuesForAUC.py --dataset airfoil --muh_fun_name NN --bias 0.85 --ntrial 10
