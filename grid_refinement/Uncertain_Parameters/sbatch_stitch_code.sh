#!/bin/bash
#SBATCH --job-name=DEM_Stitch
#SBATCH --output=results_%j.out
#SBATCH --error=results_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10M
#SBATCH --time=00:05:00

#SBATCH --partition=pbatch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=spullen@ucsd.edu

cd $SLURM_SUBMIT_DIR

echo "Python: $(which python3)" 

python3 stitching_tasks.py

# terminal command: sbatch --dependency=afterok:<first job id> sbatch_stitch_code.sh

