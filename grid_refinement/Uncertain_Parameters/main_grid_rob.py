import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective                            # importing objective function file
import os                                   # to get number of cpus
import time                                 # to track computational time
from datetime import datetime

"""
!!!!ONLY USE ON HPC BECAUSE OF MULTINODAL STRUCTURE!!!!
"""

############################################################
# Defining function for parallelization that each CPU will run
############################################################
def run_obj(i, j, c2, c3, n_samples):
    """
    This is what is passed to each cpu to run in parallel
    -----
    inputs
    -----
    i = index of c2 in the c2_sweep array
    j = index of c3 in the c2_sweep array
    c2, c3 = spring equation coefficients
    n_samples = number of realizations of the random variables (RVs) (V/V0, M/M0, zeta) or the length of stochastic_info
    -----
    outputs
    -----
    i, j =  are the indicies passing through for analysis after
    KE_cost = KE_avgs+KE_stds. what we are trying to minimize
    KE_avgs = average KE_ratio for all realizations of the RVs
    KE_stds = standard deviation of KE_ratios for all realization of the RVs
    KE_ratios = all KE_ratio's for every realization of the RVs
    """
    # if negative strain energy return with all nan
    if c3 < 2/9*c2**2:
        KE_avgs = np.nan
        KE_stds = np.nan
        KE_cost = np.nan
        return i, j, KE_cost, KE_avgs, KE_stds, np.full(n_samples, np.nan)
    
    # running the objective function
    non_spring_info = [c2,c3]
    [KE_cost, KE_avgs, KE_stds, KE_ratios] = objective.objective(non_spring_info, stochastic_info, n_samples)

    return i, j, KE_cost, KE_avgs, KE_stds, KE_ratios

############################################################
# making stochastic_info global so it doesn't have to be assigned to each CPU every time
############################################################
def init_worker(shared_data):                           
        global stochastic_info
        stochastic_info = shared_data

############################################################
# Grabing the Number of CPU's available
############################################################
def get_n_cores():
    n_cores = os.environ.get("SLURM_NTASKS")                # cores from SLURM (HPC)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',0))
    n_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT",1))

    return int(n_cores), task_id, n_jobs

############################################################
# MAIN
############################################################
if __name__ == '__main__':
    tic = time.perf_counter()                               # counts seconds

    n_cores, task_id, n_jobs = get_n_cores()                # get number of cores and tasks and jobs for multinodal computing
    ############################################################
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 1
    seed = 13510249453205735037716673912871003318           # seed for random number replication
    rng = np.random.default_rng(seed=seed)
    
    M_up = 0.45 + 0.015
    M_low = 0.45 - 0.015
    M_normal = rng.uniform(M_low, M_up, n_samples)          # varying inputs
    #M_normal = [0.05]                                      #single input normalized mass M/M0

    V_up = 0.8+0.3
    V_low = 0.8-0.3
    V_normal = rng.uniform(V_low, V_up, n_samples)          # varing inputs
    #V_normal = [1]                                         # single input normalized velocity V/V0

    Z_up = 0.15+0.01
    Z_low = 0.15-0.01
    zeta_sweep = rng.uniform(Z_low,Z_up,n_samples)          # varing inputs
    #zeta_sweep = [0.01]                                    # single input
    
    stochastic_info = [M_normal, V_normal, zeta_sweep]

    ############################################################
    # Defining C2 C3 grid coarseness
    ############################################################
    refinement_factor = (500/20)**(1/3)                                # so max c_nums = 500 after 4 steps (i=0:3)
    coarse = 20
    n_steps = 4
    c_nums = [int(np.round(coarse*(refinement_factor**i))) for i in range(n_steps)]  

    ############################################################
    # run objective function
    ############################################################
    with Pool(n_cores, initializer=init_worker, initargs=(stochastic_info,)) as pool:                              

        for k in range(len(c_nums)):
            c_num = c_nums[k]
            c2_sweep = np.linspace(-10,-1,c_num)                   # can change ranges
            c3_sweep = np.linspace(1,15,c_num)                     # can change ranges

            c2_slice = np.array_split(c2_sweep, n_jobs)[task_id]
            c2_indices = np.array_split(c_nums, n_jobs)[task_id]
            
            ############################################################
            # empty results arrays
            ############################################################
            KE_avgs   = np.zeros((c_num,c_num))
            KE_stds   = np.zeros((c_num,c_num))
            KE_cost   = np.zeros((c_num,c_num))
            KE_ratios = np.zeros((c_num,c_num, n_samples))

            tasks = [                                               # defining what is being parallelized over
                (i_global, j_local, slice[i_global], c3_sweep[j_local], n_samples) 
                for i_global in c2_indices
                for j_local in range(c_num)
            ]

            results = pool.starmap(run_obj, tasks)

            for i_global, j_local, cost, avg, std, ratio in results:             # getting the results
                KE_cost[i_global,j_local]   = cost
                KE_avgs[i_global,j_local]   = avg
                KE_stds[i_global,j_local]   = std
                KE_ratios[i_global,j_local] = ratio

    ############################################################
    # Saving data
    ############################################################
    toc = time.perf_counter()                                # getting final time
    print(f"runtime = {toc-tic:.3f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # saving the date and time
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # saving all desired outputs
    np.save(os.path.join(save_dir, f"KE_avgs_ref{k}_slice{task_id}.npy"),   KE_avgs)
    np.save(os.path.join(save_dir, "total_KE_stds.npy"),   KE_stds)
    np.save(os.path.join(save_dir, "total_KE_cost.npy"),   KE_cost)
    np.save(os.path.join(save_dir, "total_KE_ratios.npy"), KE_ratios)

    # saving coefficient options
    np.save(os.path.join(save_dir, "c_nums.npy"),  c_nums)

    # printing all necessary info to the .out file
    print(f"timestamp: {timestamp}")
    print(f"n_cores: {n_cores}")
    print(f"n_nodes: {n_jobs}")
    print(f"task_id: {task_id}")
    print(f"n_samples: {n_samples}")
    