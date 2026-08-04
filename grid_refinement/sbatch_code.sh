#!/bin/bash
#SBATCH --job-name=DEM_Python_grid_refinement
#SBATCH --output=results_%j.out
#SBATCH --error=results_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=pbatch
#SBATCH --mail-type=END
#SBATCH --mail-user=spullen@ucsd.edu

cd $SLURM_SUBMIT_DIR

echo "Cores: $SLURM_NTASKS"
echo "Python: $(which python3)" 

python3 main_grid_rob.py 
