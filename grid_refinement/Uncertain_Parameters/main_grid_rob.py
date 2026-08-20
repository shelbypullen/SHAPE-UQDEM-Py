
import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective                            # importing objective function file
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
def run_obj(i, j, c2, c3, n_samples):
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
    # if negative strain energy return with all nan
    if c3 < 2/9*c2**2:
        KE_avgs = np.nan
        KE_stds = np.nan
        KE_cost = np.nan
        return i, j, KE_cost, KE_avgs, KE_stds, np.full(n_samples, np.nan)
    
    # running the objective function
    non_spring_info = [c2,c3]
    [KE_cost, KE_avgs, KE_stds, KE_ratios] = objective.objective(non_spring_info, 
                                                                 stochastic_info, n_samples)
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
    save_name = f"results_{timestamp}_job-{job_id}"

    if len(sys.argv) > 1:
        save_dir = sys.argv[1]                             # second arg in sbatch script command line   
        print("pulling previous results from given directory")
    else:
        save_dir = os.path.join(script_dir, save_name)     # creating new results folder if first try run
        os.makedirs(save_dir, exist_ok=True)

        print(f"creating new directory for results for job id {job_id}")


    ############################################################
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 1000
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
    refinement_factor = (500/20)**(1/3)                     # so max c_nums = 500 after 4 steps (i=0:3)
    coarse = 20
    n_ref_steps = 1
    c_nums = [int(np.round(coarse*(refinement_factor**i))) 
              for i in range(n_ref_steps)]  

    # initializing the total KE arrays that will be stitched together later
    total_KE_cost   = np.empty(n_ref_steps, dtype=object)
    total_KE_avgs   = np.empty(n_ref_steps, dtype=object)
    total_KE_stds   = np.empty(n_ref_steps, dtype=object)
    total_KE_ratios = np.empty(n_ref_steps, dtype=object)

    ############################################################
    # run objective function
    ############################################################
    with Pool(n_cores, initializer=init_worker, 
              initargs=(stochastic_info,)) as pool:    
                                  
        for k in range(len(c_nums)):
            ############################################################
            # creating coefficient arrays
            ############################################################
            c_num = c_nums[k]
            c2_sweep = np.linspace(-10,-1,c_num)             # can change ranges
            c3_sweep = np.linspace(1,15,c_num)               # can change ranges

            c2_slice = np.array_split(c2_sweep, n_jobs)[task_id]
            c2_indices = np.array_split(np.arange(c_num), n_jobs)[task_id]

            ############################################################
            # empty results arrays
            ############################################################
            KE_avgs   = np.zeros((c_num,c_num))
            KE_stds   = np.zeros((c_num,c_num))
            KE_cost   = np.zeros((c_num,c_num))
            KE_ratios = np.zeros((c_num,c_num, n_samples))

            ############################################################
            # checking if this has already been computed
            ############################################################
            check_file_path = os.path.join(save_dir, f"check_ref_{k}_task_{task_id}.npy")
            if os.path.exists(check_file_path):
                print(f"check found existing results for refinement level {k} and task {task_id}: skipping")
                checkpoint = np.load(check_file_path, allow_pickle=True).item()
                KE_cost    = checkpoint['KE_cost']
                KE_avgs    = checkpoint['KE_avgs']
                KE_stds    = checkpoint['KE_stds']
                KE_ratios  = checkpoint['KE_ratios']

                # saving to total arrays as well
                total_KE_cost[k]   = KE_cost
                total_KE_avgs[k]   = KE_avgs
                total_KE_stds[k]   = KE_stds
                total_KE_ratios[k] = KE_ratios
                continue

            ############################################################
            # Running parallel job
            ############################################################
            tasks = [                                               # defining what is being parallelized over
                (i_global, j, c2_slice[i_local], c3_sweep[j], n_samples)
                for i_local, i_global in enumerate(c2_indices)
                for j in range(c_num)
            ]

            results = pool.starmap(run_obj, tasks)

            #saving results
            for i_global, j_local, cost, avg, std, ratio in results:             # getting the results
                KE_cost[i_global,j_local]   = cost
                KE_avgs[i_global,j_local]   = avg
                KE_stds[i_global,j_local]   = std
                KE_ratios[i_global,j_local] = ratio

            ############################################################
            # saving checkpoint
            ############################################################
            checkpoint = {
                'KE_cost'   : KE_cost,
                "KE_avgs"   : KE_avgs,
                "KE_stds"   : KE_stds,
                "KE_ratios" : KE_ratios
            }
            np.save(check_file_path, checkpoint)

            # saving to total arrays as well
            total_KE_cost[k]   = KE_cost
            total_KE_avgs[k]   = KE_avgs
            total_KE_stds[k]   = KE_stds
            total_KE_ratios[k] = KE_ratios
    
    ############################################################
    # Saving data
    ############################################################
    toc = time.perf_counter()                                # getting final time
    print(f"runtime = {toc-tic:.3f}")

    # saving all desired outputs with task id for later identification
    if n_jobs > 1:
        np.save(os.path.join(save_dir, f"KE_avgs_slice_{task_id}.npy"),   total_KE_avgs)
        np.save(os.path.join(save_dir, f"KE_stds_slice_{task_id}.npy"),   total_KE_stds)
        np.save(os.path.join(save_dir, f"KE_cost_slice_{task_id}.npy"),   total_KE_cost)
        np.save(os.path.join(save_dir, f"KE_ratios_slice_{task_id}.npy"), total_KE_ratios)
    else:
        np.save(os.path.join(save_dir, "total_KE_avgs.npy"),   total_KE_avgs)
        np.save(os.path.join(save_dir, "total_KE_stds.npy"),   total_KE_stds)
        np.save(os.path.join(save_dir, "total_KE_cost.npy"),   total_KE_cost)
        np.save(os.path.join(save_dir, "total_KE_ratios.npy"), total_KE_ratios)

    # saving coefficient options if another task hasn't done it yet
    if not os.path.exists(os.path.join(save_dir, "c_nums.npy")):
        np.save(os.path.join(save_dir, "c_nums.npy"),  c_nums)

    # printing all necessary info to the .out file
    print(f"timestamp: {timestamp}")
    print(f"n_cores: {n_cores}")
    print(f"n_nodes: {n_jobs}")
    print(f"task_id: {task_id}")
    print(f"n_samples: {n_samples}")

    ############################################################
    # deleting checkpoints to save space now that all info is combined
    ############################################################
    for k in range(n_ref_steps):
        check_file_path = os.path.join(save_dir, f"check_ref_{k}_task_{task_id}.npy")
        if os.path.exists(check_file_path):
            os.remove(check_file_path)
    
    print(f"Task {task_id} completed and checkpoints cleared")
