#!/bin/bash
#SBATCH --job-name=200_DEM_MC
#SBATCH --output=200results_main_%A_%a.out
#SBATCH --error=200results_main_%A_%a.err

#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=112
#SBATCH --mem=5G
#SBATCH --time=06:30:00

#SBATCH --partition=pbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=spullen@ucsd.edu

cd $SLURM_SUBMIT_DIR

echo "Python: $(which python3)" 
echo "Cores: $SLURM_NTASKS"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

python3 200main_grid_rob.py #include directory name as second arg if previously failed


