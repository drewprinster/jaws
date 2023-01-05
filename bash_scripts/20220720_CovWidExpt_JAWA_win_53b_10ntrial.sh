#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 
#SBATCH --mem=50G
conda activate PredictorAudit
python3 ../20220716_run_JAWA.py --d wine --b 0.53 --ntrial 10 --L2_lambda 8.0