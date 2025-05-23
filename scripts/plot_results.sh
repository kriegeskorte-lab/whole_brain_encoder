#!/bin/bash
#SBATCH -A nklab           # Set Account name
#SBATCH --job-name=plot_results
#SBATCH -c 4
#SBATCH -t 0-01:00              # Runtime in D-HH:MM
#XX SBATCH --gres=gpu:1
#SBATCH --mail-user=eh2976@columbia.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_logs/plot_results_%j.out

cd /engram/nklab/algonauts/ethan/whole_brain_encoder/
source /home/eh2976/.bashrc

conda activate xformers
ml load gcc/10.4 # needed for torch compile to work properly

# python eval_model.py --subj $@

conda activate pycortex

python plot_run_results.py $@