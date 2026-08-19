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
all_results = os.path.join(script_dir, "results_2026-08-18_job-7166688")
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

failed_task_ids = [10,11,12,13]
n_sub = 4
n_jobs = 15

# load all sub slices once
cost_sub   = {}
avgs_sub   = {}
stds_sub   = {}
ratios_sub = {}

n_samples = 1000

for task_id in failed_task_ids:
    for sub_id in range(n_sub):
        cost_sub[task_id, sub_id]   = np.load(os.path.join(latest_results_dir, f"KE_cost_slice_{task_id}_{sub_id}.npy"),   allow_pickle=True)
        avgs_sub[task_id, sub_id]   = np.load(os.path.join(latest_results_dir, f"KE_avgs_slice_{task_id}_{sub_id}.npy"),   allow_pickle=True)
        stds_sub[task_id, sub_id]   = np.load(os.path.join(latest_results_dir, f"KE_stds_slice_{task_id}_{sub_id}.npy"),   allow_pickle=True)
        ratios_sub[task_id, sub_id]   = np.load(os.path.join(latest_results_dir, f"KE_ratios_slice_{task_id}_{sub_id}.npy"),   allow_pickle=True)

for task_id in failed_task_ids:
    total_KE_cost = np.empty(n_ref_steps, dtype=object)
    total_KE_avgs = np.empty(n_ref_steps, dtype=object)
    total_KE_stds = np.empty(n_ref_steps, dtype=object)
    total_KE_ratios = np.empty(n_ref_steps, dtype=object)

    for k in range(n_ref_steps):
        c_num = int(c_nums[k])

        #initializing within each task_id
        KE_cost_full   = np.full((c_num, c_num), np.nan)
        KE_avgs_full   = np.full((c_num, c_num), np.nan)
        KE_stds_full   = np.full((c_num, c_num), np.nan)
        KE_ratios_full = np.full((c_num, c_num, n_samples), np.nan)

        c2_task_indicies = np.array_split(np.arange(c_num), n_jobs)[task_id]

        for sub_id in range(n_sub):
            c2_sub_indices = np.array_split(c2_task_indicies, n_sub)[sub_id]

            KE_cost_full[c2_sub_indices, :] = cost_sub[task_id, sub_id][k][c2_sub_indices, :]
            KE_avgs_full[c2_sub_indices, :] = avgs_sub[task_id, sub_id][k][c2_sub_indices, :]
            KE_stds_full[c2_sub_indices, :] = stds_sub[task_id, sub_id][k][c2_sub_indices, :]
            KE_ratios_full[c2_sub_indices, :, :] = ratios_sub[task_id, sub_id][k][c2_sub_indices, :,:]

        total_KE_cost[k] = KE_cost_full
        total_KE_avgs[k] = KE_avgs_full
        total_KE_stds[k] = KE_stds_full
        total_KE_ratios[k] = KE_ratios_full

    np.save(os.path.join(latest_results_dir, f"KE_cost_slice{task_id}.npy"),   total_KE_cost)
    np.save(os.path.join(latest_results_dir, f"KE_avgs_slice{task_id}.npy"),   total_KE_avgs)
    np.save(os.path.join(latest_results_dir, f"KE_stds_slice{task_id}.npy"),   total_KE_stds)
    np.save(os.path.join(latest_results_dir, f"KE_ratios_slice{task_id}.npy"),   total_KE_ratios)
    print(f"task {task_id} saved as KE_cost_slice{task_id}.npy")

toc = time.perf_counter()
print(f"runtime = {toc-tic}")
    