#!/bin/bash
#SBATCH --account=def-jpineau
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=01-00:00
#SBATCH --mail-user=ruo.tao@mail.mcgill.ca
#SBATCH --mail-type=ALL

source ../venv/bin/activate
python ../main.py -e ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll

