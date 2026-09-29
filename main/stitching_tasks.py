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
all_results = os.path.join(script_dir, "results_2026*")
folders = glob.glob(all_results)

if not folders:
    raise FileNotFoundError("No Result Folder Found")

latest_results_dir = max(folders, key=os.path.getmtime)     # os.path.getmtime gives the most recent 
                                                            # time modified as what is being maximized
print("results directory: ", latest_results_dir)
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
opt_KE_ratios = np.empty(n_ref_steps, dtype=object)

#getting number of jobs e.g. how many different cost files are there
all_tasks = os.path.join(latest_results_dir, "KE_ratios_slice_*.npy")
task_files = glob.glob(all_tasks)
n_jobs = len(task_files)                                    # need number of jobs

ratios_slice = [np.load(os.path.join(latest_results_dir, f"KE_ratios_slice_{task_id}.npy"), allow_pickle=True) 
                for task_id in range(n_jobs)]

n_samples = ratios_slice[0][0].shape[2]                     # need number of MC samples
n_per_task = [np.array_split(np.arange(n_samples), n_jobs)[task_id]
                      for task_id in range(n_jobs)]

opt_c2s = np.zeros(n_ref_steps)
opt_c3s = np.zeros(n_ref_steps)
opt_KEs = np.zeros((n_ref_steps, 3))                                  # each row = [opt_cost, opt KE_avg, opt_std]

############################################################
# stitching everything together
############################################################
for k in range(n_ref_steps):
    c_num = int(c_nums[k])
    c2_sweep = np.linspace(-8,-1,c_num)
    c3_sweep = np.linspace(1,15,c_num)

    #initializing within each task_id
    KE_cost_full   = np.full((c_num, c_num), np.nan)
    KE_avgs_full   = np.full((c_num, c_num), np.nan)
    KE_stds_full   = np.full((c_num, c_num), np.nan)
    KE_ratios_full = np.full((c_num, c_num, n_samples), np.nan)

    # stitching
    for task_id in range(n_jobs):
        KE_ratios_full[:,:,n_per_task[task_id]] = ratios_slice[task_id][k][:,:,n_per_task[task_id]]

    for i in range(len(c2_sweep)):
        for j in range(len(c3_sweep)):
            c2 = c2_sweep[i]
            c3 = c3_sweep[j]

            if c3 < 2/9*c2**2:
                # KE_avgs = np.nan
                # KE_stds = np.nan
                # KE_cost = np.nan
                continue

            ############################################################
            # Calculate Cost Function
            ############################################################
            # get the normalization variables from c2 = -4.5, c3 = 8 (center of the grid) for n = 10_000

            # script_dir = os.path.dirname(__file__)
            # norm_directory = os.path.join(script_dir, "normalization_variables")
            # mu0 = np.load(os.path.join(norm_directory, "mean_0.npy"), allow_pickle=True)
            # std0 = np.load(os.path.join(norm_directory, "std_0.npy"), allow_pickle=True)

            mu0 = 0.3299092975807557
            std0 = 0.16766085514773352

            # define weights for J = w1*E(KE) + w2*sqrt(V(KE)), w1+w2=1
            w1 = 0.5
            w2 = 0.5

            if w1 + w2 != 1:
                raise ValueError("w1 + w2 !=1 in cost function")

            # start calculating
            KE_avgs_full[i,j] = np.average(KE_ratios_full[i,j,:])
            
            # only taking upper standard deviation into account because thats what we want to minimize
            KE_upper_vals = KE_ratios_full[i,j,:][KE_ratios_full[i,j,:] > KE_avgs_full[i,j]]
            KE_stds_full[i,j] = 0
            if n_samples >1:
                KE_stds_full[i,j] = np.std(KE_upper_vals, ddof=1)

            if KE_stds_full[i,j] >0:
                KE_cost_full[i,j] = w1*KE_avgs_full[i,j]/mu0 + w2*KE_stds_full[i,j]/std0
            else:
                KE_cost_full[i,j] = KE_avgs_full[i,j]

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

    #get optimal KE_ratios so I can delete unneccesary data
    #opt_KE_ratios[k] = KE_ratios_full[c2_idx,c3_idx]

############################################################
# saving complete arrays
############################################################
np.save(os.path.join(latest_results_dir, "total_KE_cost.npy"), total_KE_cost)
np.save(os.path.join(latest_results_dir, "total_KE_avgs.npy"), total_KE_avgs)
np.save(os.path.join(latest_results_dir, "total_KE_stds.npy"), total_KE_stds)
np.savez_compressed(os.path.join(latest_results_dir, "opt_KE_ratios.npz"), opt_KE_ratios)
np.savez_compressed(os.path.join(latest_results_dir, "total_KE_ratios.npz"), total_KE_ratios)
print("All files saved")

# printing all necessary info to the .out file
for k in range(n_ref_steps):
   print(f"refinement level {k}: c2 = {opt_c2s[k]:.4f}, c3 = {opt_c3s[k]:.4f}, ")
   print(f"KE_cost = {opt_KEs[k,0]:.4f}, KE_avg = {opt_KEs[k,1]:.4f}, KE_std = {opt_KEs[k,2]:.4f}")


# delete slices:
for k in range(n_jobs):
    ratios_slice_file_path = os.path.join(latest_results_dir, f"KE_ratios_slice_{k}.npy")
    if os.path.exists(ratios_slice_file_path):
        os.remove(ratios_slice_file_path)
    print(f"task {k} slices deleted")

toc = time.perf_counter()
print(f"runtime = {toc-tic}")
    
