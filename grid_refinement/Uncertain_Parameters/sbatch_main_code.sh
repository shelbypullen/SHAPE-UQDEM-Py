#!/bin/bash
#SBATCH --job-name=DEM_UQ_arrays
#SBATCH --output=results_main_%A_%a.out
#SBATCH --error=results_main_%A_%a.err

#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=5G
#SBATCH --time=00:05:00

#SBATCH --partition=pbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=spullen@ucsd.edu

cd $SLURM_SUBMIT_DIR

echo "Python: $(which python3)" 
echo "Cores: $SLURM_NTASKS"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

python3 main_grid_rob.py #include directory name as second arg if previously failed


