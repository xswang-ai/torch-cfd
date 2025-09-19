#!/bin/bash
# This script produces 5.2k training, 1.3k valid, and 1.3k test trajectories of the Navier-Stokes dataset.

#SBATCH --time=20:00:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)

# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
module load parallel python
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate

# python3 data_gen_fno.py --num-samples 500 --batch-size 256 --grid-size 256 --subsample 2 --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --visc 1e-3
python3 data_gen_McWilliams2d.py --num-samples 5000 --batch-size 256 --grid-size 256 --subsample 2 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"