import numpy as np
import os
import glob

script_dir = os.path.dirname(__file__)
all_results = os.path.join(script_dir, "results_2026-08-12_job-1")
folders = glob.glob(all_results)
print(folders)
path = os.path.join(all_results, "check_ref_0_task_0.npy")
print(np.load(path, allow_pickle=True))