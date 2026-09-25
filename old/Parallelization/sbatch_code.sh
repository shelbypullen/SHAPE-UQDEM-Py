#!/bin/bash
#SBATCH --job-name=20_by_20_DEM_MC
#SBATCH --output=20results_%j.out
#SBATCH --error=20results_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=pbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=spullen@ucsd.edu

cd $SLURM_SUBMIT_DIR

echo "Cores: $SLURM_NTASKS"
echo "Python: $(which python3)" 

python3 main_grid_rob.py 
