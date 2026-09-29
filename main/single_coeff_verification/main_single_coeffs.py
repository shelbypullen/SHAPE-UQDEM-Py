import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective_copy                            # importing objective function file
import os                                   # to get number of cpus
import time                                 # to track computational time
from datetime import datetime
import sys

"""
built for HPC but can handle running on personal computer 
!!! will take like a year !!!
""" 
############################################################
# Defining function for parallelization that each CPU will run
############################################################
def run_obj(c2, c3, k, M_normal, V_normal, zeta_sweep):
    """
    This is what is passed to each node to run in parallel and the with Pool () 
        distributes the objective function to run on each cpu
    -----
    inputs
    -----
    i = index of c2 in the c2_sweep array
    j = index of c3 in the c2_sweep array
    c2, c3 = spring equation coefficients
    n_samples = number of realizations of the random variables (RVs) (V/V0, M/M0, zeta) 
        or the length of stochastic_info
    -----
    outputs
    -----
    i, j =  are the indicies passing through for analysis after
    KE_cost = KE_avgs+KE_stds. what we are trying to minimize
    KE_avgs = average KE_ratio for all realizations of the RVs
    KE_stds = standard deviation of KE_ratios for all realization of the RVs
    KE_ratios = all KE_ratio's for every realization of the RVs
    """
    print(k)
    # if negative strain energy return with all nan
    if c3 < 2/9*c2**2:
        return k, np.full(n_samples, np.nan)

    # running the objective function
    non_spring_info = [c2,c3]
    stochastic_info = [M_normal, V_normal, zeta_sweep]
    KE_ratios = objective_copy.objective(non_spring_info, stochastic_info, 1)

    return k, KE_ratios

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
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',0))  # which nodal task is it defaults to 0 for 1st index
    n_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT",1))# how many nodes/tasks are there defaults to 1 for 1 job/node
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID",1))
    
    if n_cores is not None:                                 # if this is an HPC Job, use number of cpus from SLURM
        return int(n_cores), task_id, n_jobs, job_id
    else:
        return max(1,os.cpu_count()-1),task_id,n_jobs,job_id# if no slurm cpu allocation (i.e. running on a local laptop) 
                                                            # use local CPU # -1 for background processes
############################################################
# MAIN
############################################################
if __name__ == '__main__':

    tic = time.perf_counter()                               # counts seconds

    n_cores, task_id, n_jobs, job_id = get_n_cores()        # get number of cores and tasks and jobs for multinodal computing

    ############################################################
    # creating results directory or grabbing it from previously failed run
    ############################################################
    timestamp = datetime.now().strftime("%Y-%m-%d")# saving the date and time
    script_dir = os.path.dirname(__file__)                 # getting current file directory
    
    ############################################################
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 10_000
    seed = 13510249453205735037716673912871003318           # seed for random number replication
    rng = np.random.default_rng(seed=seed)
    
    M_up = 0.045 + 0.015
    M_low = 0.045 - 0.015
    M_normal = rng.uniform(M_low, M_up, n_samples)          # varying inputs
    #M_normal = [0.05]                                      #single input normalized mass M/M0

    V_up = 0.8+0.3
    V_low = 0.8-0.3
    V_normal = rng.uniform(V_low, V_up, n_samples)          # varing inputs
    #V_normal = [1]                                         # single input normalized velocity V/V0

    Z_up = 0.015+0.01
    Z_low = 0.015-0.01
    zeta_sweep = rng.uniform(Z_low,Z_up,n_samples)          # varing inputs
    #zeta_sweep = [0.01]                                    # single input
    
    stochastic_info = [M_normal, V_normal, zeta_sweep]

    ############################################################
    # Defining C2 C3 grid coarseness
    ############################################################
    refinement_factor = (500/20)**(1/3)                     # so max c_nums = 500 after 4 steps (i=0:3)
    coarse = 100
    n_ref_steps = 1
    c_nums = [int(np.round(coarse*(refinement_factor**i))) 
              for i in range(n_ref_steps)]  

    # initializing the total KE arrays that will be stitched together later
    total_KE_ratios = np.empty(n_ref_steps, dtype=object)

    total_time = np.empty(n_ref_steps, dtype=object)
    ############################################################
    # run objective function
    ############################################################
    with Pool(n_cores, initializer=init_worker, 
              initargs=(stochastic_info,)) as pool:    
                                  
        for k in range(len(c_nums)):
            ############################################################
            # creating coefficient arrays
            ############################################################
            #c_num = c_nums[k]
            #c2_sweep = np.linspace(-8,-1,c_num)             # can change ranges
            #c3_sweep = np.linspace(1,15,c_num)               # can change ranges

            c_num = 1
            c2_sweep = -4.5
            c3_sweep = 8

            print(f"c2 = {c2_sweep}, c3 = {c3_sweep}")

            ############################################################
            # empty results arrays
            ############################################################
            KE_ratios = np.zeros(n_samples)

            ############################################################
            # Running parallel job
            ############################################################
            tasks = [                                               # defining what is being parallelized over
                (c2_sweep, c3_sweep, k, M_normal[k], V_normal[k], zeta_sweep[k])
                for k in range(n_samples)
            ]

            results = pool.starmap(run_obj, tasks)

            #saving results
            for k, ratio in results:             # getting the results
                KE_ratios[k] = ratio

    ############################################################
    # Saving data
    ############################################################
    toc = time.perf_counter()                                # getting final time
    print(f"runtime = {toc-tic:.3f}")

    ############################################################
    # Calculate Cost Function
    ############################################################
    KE_avg = np.average(KE_ratios)
    
    # only taking upper standard deviation into account because thats what we want to minimize
    KE_upper_vals = KE_ratios[KE_ratios > KE_avg]
    KE_std = 0
    if n_samples >1:
        KE_std = np.std(KE_upper_vals, ddof=1)

    if KE_std >0:
        J_function = KE_avg + KE_std
    else:
        J_function = KE_avg

    print(f"KE_mean = {KE_avg}, KE_std = {KE_std}")

    np.save(os.path.join(script_dir, "mean_0.npy"), KE_avg)
    np.save(os.path.join(script_dir, "std_0.npy"), KE_std)
    np.save(os.path.join(script_dir, "total_0.npy"), KE_ratios)