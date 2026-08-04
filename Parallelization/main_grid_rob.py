import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective                            # importing objective function file
import os                                   # to get number of cpus
import time                                 # to track computational time
from datetime import datetime

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

    if n_cores is not None:                                 # if this is an HPC Job, use number of cpus from SLURM
        return int(n_cores)
    else:
        return max(1, os.cpu_count() - 1)                   # if no slurm cpu allocation (i.e. running on a local laptop) use local CPU # -1 for background processes

############################################################
# MAIN
############################################################
if __name__ == '__main__':
    tic = time.perf_counter()                               # counts seconds

    n_cores = get_n_cores()                                 # get number of cores
    ############################################################
    # Defining C2 C3 space - CAN CHANGE c2_num, c3_num FOR HIGHER PRECISION
    ############################################################
    c2_num = 20                                 
    c3_num = 20
    c2_sweep = np.linspace(-10,-1,c2_num)                   # can change ranges
    c3_sweep = np.linspace(1,15,c3_num)                     # can change ranges


    ############################################################
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 1
    rng = np.random.default_rng()
    M_up = 0.45 + 0.015
    M_low = 0.45 - 0.015
    #M_normal = rng.uniform(M_low, M_up, n_samples)         # varying inputs
    N = 20
    M0_impactor = N/2
    M_normal = [0.05]                                       #single input

    V_up = 0.8+0.3
    V_low = 0.8-0.3
    #V_normal = rng.uniform(V_low, V_up, n_samples)         # varing inputs
    V_normal = [1]                                          # single input

    Z_up = 0.15+0.01
    Z_low = 0.15-0.01
    #zeta_sweep = rng.uniform(Z_low,Z_up,n_samples)         # varing inputs
    zeta_sweep = [0.01]                                     # signle input
    stochastic_info = [M_normal, V_normal, zeta_sweep]

    ############################################################
    # empty results arrays
    ############################################################
    KE_avgs = np.zeros((c2_num,c3_num))
    KE_stds = np.zeros((c2_num,c3_num))
    KE_cost = np.zeros((c2_num,c3_num))
    KE_ratios = np.zeros((c2_num,c3_num, n_samples))

    ############################################################
    # run objective function
    ############################################################
    tasks = [                                               # defining what is being parallelized over
        (i, j, c2_sweep[i], c3_sweep[j], n_samples) 
        for i in range(c2_num)
        for j in range(c3_num)
    ]

    with Pool(n_cores, initializer=init_worker, initargs=(stochastic_info,)) as pool:
        results = pool.starmap(run_obj, tasks)

    for i, j, cost, avg, std, ratio in results:             # getting the results
        KE_cost[i,j] = cost
        KE_avgs[i,j] = avg
        KE_stds[i,j] = std
        KE_ratios[i,j] = ratio
    ############################################################
    # get optimal values
    ############################################################
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(KE_cost), KE_cost.shape)      # where the optimal spring coefficients are

    toc = time.perf_counter()                                # getting final time
    print(f"runtime = {toc-tic:.3f}")

    ############################################################
    # Saving data
    ############################################################
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # saving the date and time
    #save_dir = f"results_{timestamp}"
    #os.makedirs(save_dir, exist_ok=True)

    # saving all desired outputs
    #np.save(os.path.join(save_dir, "KE_avgs.npy"),   KE_avgs)
    #np.save(os.path.join(save_dir, "KE_stds.npy"),   KE_stds)
    #np.save(os.path.join(save_dir, "KE_cost.npy"),   KE_cost)
    #p.save(os.path.join(save_dir, "KE_ratios.npy"), KE_ratios)

    # saving coefficient options
    #np.save(os.path.join(save_dir, "c2_sweep.npy"),  c2_sweep)
    #np.save(os.path.join(save_dir, "c3_sweep.npy"),  c3_sweep)

    # printing all necessary info to the .out file
    print(f"timestamp: {timestamp}")
    print([f"c2_num: {c2_num}", f"c3_num: {c3_num}"])
    print(f"n_cores: {n_cores}")
    print(f"n_samples: {n_samples}")
    print([f"best c2: {c2_sweep[c2_idx]}", f"best c3: {c3_sweep[c3_idx]}"])
    print(f"min KE_avg: {np.nanmin(KE_avgs)}")
    print(f"min KE_std: {np.nanmin(KE_stds)}")
    print(f"min KE_cost: {np.nanmin(KE_cost)}")