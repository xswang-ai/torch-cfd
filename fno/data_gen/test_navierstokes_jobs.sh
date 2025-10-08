#!/bin/bash
# This script produces 5.2k training, 1.3k valid, and 1.3k test trajectories of the Navier-Stokes dataset.

#SBATCH --time=00:30:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)

# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
module load parallel python
module load tensorflow/2.16.1-pip-py312-cuda122
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate

# python3 data_gen_fno.py --num-samples 500 --batch-size 256 --grid-size 256 --subsample 2 --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --visc 1e-3

## McWilliams 2d Re=1000
# training
# python3 data_gen_McWilliams2d.py --num-samples 5000 --batch-size 256 --grid-size 256 --subsample 2 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"
# # validation
# python3 data_gen_McWilliams2d.py --num-samples 256 --batch-size 256 --grid-size 256 --subsample 2 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --seed 0
# # test
# python3 data_gen_McWilliams2d.py --num-samples 500 --batch-size 256 --grid-size 256 --subsample 2 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --seed 500

## McWilliams 2d Re=100
# training
# python3 data_gen_McWilliams2d.py --num-samples 1 --batch-size 256 --grid-size 256 --subsample 2 --visc 1e-2 --dt 1e-3 --time 102 --time-warmup 18 --num-steps 100 --diam "2*torch.pi"


# McWilliams 2d Re=5000
# training
python data_gen_McWilliams2d.py --num-samples 1 --batch-size 256 --grid-size 256 --subsample 2 --Re 5e3 --dt 5e-4 --time 100 --time-warmup 18 --num-steps 100 --diam "2*torch.pi"
# # validation
# python data_gen_McWilliams2d.py --num-samples 256 --batch-size 256 --grid-size 512 --subsample 4 --Re 5e3 --dt 5e-4 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --seed 0
# # # test
# python data_gen_McWilliams2d.py --num-samples 500 --batch-size 256 --grid-size 512 --subsample 4 --Re 5e3 --dt 5e-4 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --seed 500