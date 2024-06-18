#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduproject
#SBATCH --nodelist=cn47
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/fharrathi/logs/experiment1_%J.out
#SBATCH --error=/home/fharrathi/logs/experiment1_%J.err
# SBATCH --mail-user=fharrathi
# SBATCH --mail-type=BEGIN,END,FAIL

### notes
# this experiment is meant to try out training the default k-nn model on the cluster

# location of repository and data
project_dir=. # assume sbatch is called from root project dir
#cifar10_folder=$project_dir/data/cifar10

source "$project_dir"/venv/bin/activate
srun python /scratch/fharrathi/virtual_environments/RUthesis/train.py  \
