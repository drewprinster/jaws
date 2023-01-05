#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20
#SBATCH --mem=100G 
python3 ../run_JAW_PredictiveValuesForAUC.py --dataset superconduct --muh_fun_name NN --bias 0.00062 --ntrial 10
