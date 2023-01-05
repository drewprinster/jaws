#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=50G 
conda activate PredictorAudit
python3 ../run_JAW_expts.py --dataset airfoil --muh_fun_name NN --bias 0.85 --ntrial 10