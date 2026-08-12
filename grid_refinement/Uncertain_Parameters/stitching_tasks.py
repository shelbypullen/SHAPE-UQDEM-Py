import numpy as np
import os
import glob
from datetime import datetime
import time

# start timer
tic = time.perf_counter()

############################################################
# getting path to most recent folder modified
############################################################
script_dir = os.path.dirname(__file__)
all_results = os.path.join(script_dir, "results_*")
folders = glob.glob(all_results)

if not folders:
    raise FileNotFoundError("No Result Folder Found")

latest_results_dir = max(folders, key=os.path.getmtime)     # os.path.getmtime gives the most recent 
                                                            # time modified as what is being maximized
############################################################
# initializing arrays to be returned and loading arrays
############################################################
c_nums_path = os.path.join(latest_results_dir, "c_nums.npy")
c_nums = np.load(c_nums_path, allow_pickle=True)
n_ref_steps = len(c_nums)

total_KE_cost = np.empty(n_ref_steps, dtype=object)
total_KE_avgs = np.empty(n_ref_steps, dtype=object)
total_KE_stds = np.empty(n_ref_steps, dtype=object)
total_KE_ratios = np.empty(n_ref_steps, dtype=object)

#getting number of jobs e.g. how many different cost files are there
all_tasks = os.path.join(latest_results_dir, "KE_avgs_slice_*.npy")
task_files = glob.glob(all_tasks)
n_jobs = len(task_files)                                    # need number of jobs

cost_slice = [np.load(os.path.join(latest_results_dir, f"KE_cost_slice_{task_id}.npy"), allow_pickle=True) 
              for task_id in range(n_jobs)]
avgs_slice = [np.load(os.path.join(latest_results_dir, f"KE_avgs_slice_{task_id}.npy"), allow_pickle=True) 
              for task_id in range(n_jobs)]
stds_slice = [np.load(os.path.join(latest_results_dir, f"KE_stds_slice_{task_id}.npy"), allow_pickle=True) 
              for task_id in range(n_jobs)]
ratios_slice = [np.load(os.path.join(latest_results_dir, f"KE_ratios_slice_{task_id}.npy"), allow_pickle=True) 
                for task_id in range(n_jobs)]

n_samples = ratios_slice[0][0].shape[2]                     # need number of MC samples

opt_c2s = np.zeros(n_ref_steps)
opt_c3s = np.zeros(n_ref_steps)
opt_KEs = np.zeros((n_ref_steps, 3))                                  # each row = [opt_cost, opt KE_avg, opt_std]
############################################################
# stitching everything together
############################################################
for k in range(n_ref_steps):
    c_num = int(c_nums[k])
    c2_sweep = np.linspace(-10,-1,c_num)
    c3_sweep = np.linspace(1,15,c_num)

    #initializing within each task_id
    KE_cost_full   = np.full((c_num, c_num), np.nan)
    KE_avgs_full   = np.full((c_num, c_num), np.nan)
    KE_stds_full   = np.full((c_num, c_num), np.nan)
    KE_ratios_full = np.full((c_num, c_num, n_samples), np.nan)

    # stitching
    for task_id in range(n_jobs):
        c2_indicies = np.array_split(np.arange(c_num), n_jobs)[task_id]
        print(c2_indicies)
        KE_cost_full[c2_indicies,:] = cost_slice[task_id][k][c2_indicies,:]
        KE_avgs_full[c2_indicies,:] = avgs_slice[task_id][k][c2_indicies,:]
        KE_stds_full[c2_indicies,:] = stds_slice[task_id][k][c2_indicies,:]
        KE_ratios_full[c2_indicies,:] = ratios_slice[task_id][k][c2_indicies,:,:]
    
    #appending
    total_KE_cost[k] = KE_cost_full
    total_KE_avgs[k] = KE_avgs_full
    total_KE_stds[k] = KE_stds_full
    total_KE_ratios[k] = KE_ratios_full

    ############################################################
    # get optimal values
    ############################################################
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(KE_cost_full), KE_cost_full.shape)      # where the optimal spring coefficients are
    opt_c2s[k] = c2_sweep[c2_idx]
    opt_c3s[k] = c3_sweep[c3_idx]
    opt_KEs[k] = [np.nanmin(KE_cost_full), KE_avgs_full[c2_idx,c3_idx], KE_stds_full[c2_idx,c3_idx]]

############################################################
# saving complete arrays
############################################################
np.save(os.path.join(latest_results_dir, "total_KE_cost.npy"), total_KE_cost)
np.save(os.path.join(latest_results_dir, "total_KE_avgs.npy"), total_KE_avgs)
np.save(os.path.join(latest_results_dir, "total_KE_stds.npy"), total_KE_stds)
np.save(os.path.join(latest_results_dir, "total_KE_ratios.npy"), total_KE_ratios)

toc = time.perf_counter()
print(f"runtime = {toc-tic}")

# printing all necessary info to the .out file
for k in range(n_ref_steps):
    print(f"refinement level {k}: c2 = {opt_c2s[k]:.4f}, c3 = {opt_c3s[k]:.4f}, ")
    print(f"KE_cost = {opt_KEs[k,0]:.4f}, KE_avg = {opt_KEs[k,1]:.4f}, KE_std = {opt_KEs[k,2]:.4f}")