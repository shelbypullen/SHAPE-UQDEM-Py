import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective                            # importing objective function file
import os                                   # to get number of cpus
import time                                 # to track computational time
from datetime import datetime
import sys
import glob

"""
built for HPC but can handle running on personal computer 
!!! will take like a year !!!
"""
############################################################
# Defining function for parallelization that each CPU will run
############################################################
def run_obj(i, j, c2, c3, n_samples_slice):
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
    tic1 = time.process_time()

    # if negative strain energy return with all nan
    if c3 < 2/9*c2**2:
        return i, j, np.full(n_samples_slice, np.nan), 0

    # running the objective function
    non_spring_info = [c2,c3]
    KE_ratios = objective.objective(non_spring_info, stochastic_info, n_samples_slice)

    toc1 = time.process_time()
    cpu_time = toc1 - tic1
    return i, j, KE_ratios, cpu_time

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
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID",12345))
    
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
    timestamp = datetime.now().strftime("%Y-%m-%d")             # saving the date and time
    script_dir = os.path.dirname(__file__)                      # getting current file directory
    save_name = f"results_{timestamp}_job-{job_id}"

    if len(sys.argv) > 1:
        save_dir = sys.argv[1]                                  # second arg in sbatch script command line   
        print("pulling previous results from given directory")
    else:
        # first see if another directory from another day with same job ID (needed for big jobs)
        if job_id > 1000:                                        # not the default job id
            existing = glob.glob(os.path.join(script_dir, f"results_*_job-{job_id}"))
            existing = [dir for dir in existing if os.path.isdir(dir)]

            if existing:                                        # if there exists a previous directory for this job
                save_dir = max(existing, key=os.path.getmtime)
                print(f"found existing directory for job: {job_id}: {save_dir}")
            else:
                save_dir = os.path.join(script_dir, save_name)  # creating new results folder if first try run
                os.makedirs(save_dir, exist_ok=True)
                print(f"creating new directory for results for job id {job_id}")
        else:
            save_dir = os.path.join(script_dir, save_name)      # creating new results folder if first try run
            os.makedirs(save_dir, exist_ok=True)
            print(f"creating new directory for results for job id {job_id}")

    # recording info for out file
    print(f"timestamp: {timestamp}")
    print(f"n_cores: {n_cores}")
    print(f"n_nodes: {n_jobs}")
    print(f"task_id: {task_id}")

    ############################################################
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 2000

    # generate RV realizations
    seed = 13510249453205735037716673912871003318               # seed for random number replication
    rng = np.random.default_rng(seed=seed)
    
    M_up = 0.045 + 0.015
    M_low = 0.045 - 0.015
    M_normal_tot = rng.uniform(M_low, M_up, n_samples)          # varying inputs
    M_normal = np.array_split(M_normal_tot, n_jobs)[task_id]    # split into task arrays for multinode
    #M_normal = [0.05]                                          #single input normalized mass M/M0

    V_up = 0.8+0.3
    V_low = 0.8-0.3
    V_normal_tot = rng.uniform(V_low, V_up, n_samples)          # varing inputs
    V_normal = np.array_split(V_normal_tot, n_jobs)[task_id]    # split into task arrays for multinode
    #V_normal = [1]                                             # single input normalized velocity V/V0

    Z_up = 0.015+0.01
    Z_low = 0.015-0.01
    zeta_sweep_tot = rng.uniform(Z_low,Z_up,n_samples)          # varing inputs
    zeta_sweep = np.array_split(zeta_sweep_tot,n_jobs)[task_id] # split into task arrays for multinode
    #zeta_sweep = [0.01]                                        # single input
    
    stochastic_info = [M_normal, V_normal, zeta_sweep]
    n_samples_indices = np.array_split(np.arange(n_samples), n_jobs)[task_id]
    n_samples_slice = len(n_samples_indices)

    # record n_samples to out 
    print(f"n_samples: {n_samples}")
    print(f"n_samples_slice: {n_samples_slice}")
    ############################################################
    # Defining C2 C3 grid coarseness
    ############################################################
    refinement_factor = (500/20)**(1/3)                         # so max c_nums = 500 after 4 steps (i=0:3)
    coarse = 200
    n_ref_steps = 1
    c_nums = [int(np.round(coarse*(refinement_factor**i))) 
              for i in range(n_ref_steps)]  

    # initializing the total KE arrays that will be stitched together later
    total_KE_ratios = np.empty(n_ref_steps, dtype=object)

    total_time = np.empty(n_ref_steps, dtype=object)

    print(f"n_ref_steps: {n_ref_steps}")
    print(f"c_nums: {c_nums}")
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
            c2_sweep = np.linspace(-8,-1,c_num)              # can change ranges
            c3_sweep = np.linspace(1,15,c_num)               # can change ranges

            ############################################################
            # empty results arrays
            ############################################################
            KE_ratios = np.full((c_num,c_num, n_samples),np.nan)

            cpu_times = np.zeros((c_num,c_num))
            ############################################################
            # checking if this has already been computed
            ############################################################
            check_file_path = os.path.join(save_dir, f"check_ref_{k}_task_{task_id}.npy")
            if os.path.exists(check_file_path):
                print(f"check found existing results for refinement level {k} and task {task_id}: skipping")
                checkpoint = np.load(check_file_path, allow_pickle=True).item()
                KE_ratios  = checkpoint['KE_ratios']
                cpu_times   = checkpoint['cpu_times']

                # saving to total arrays as well
                total_KE_ratios[k] = KE_ratios
                total_time[k]      = cpu_times
                continue
            
            ############################################################
            # Running parallel job
            ############################################################
            tasks = [                                                   # defining what is being parallelized over
                (i, j, c2_sweep[i], c3_sweep[j], n_samples_slice)
                for i in range(c_num)
                for j in range(c_num)
            ]

            results = pool.starmap(run_obj, tasks)

            #saving results
            for i, j, ratio, cpu_time in results:       # getting the results
                KE_ratios[i,j,n_samples_indices] = ratio
                cpu_times[i,j]  = cpu_time

            ############################################################
            # saving checkpoint
            ############################################################
            checkpoint = {
                "KE_ratios" : KE_ratios,
                "cpu_times"  : cpu_times
            }
            np.save(check_file_path, checkpoint)

            # saving to total arrays as well
            total_KE_ratios[k] = KE_ratios
            total_time[k]      = cpu_times
    
    ############################################################
    # Saving data
    ############################################################
    toc = time.perf_counter()                                   # getting final time
    print(f"runtime = {toc-tic:.3f}")

    # saving all desired outputs with task id for later identification
    if n_jobs > 1:
        np.save(os.path.join(save_dir, f"KE_ratios_slice_{task_id}.npy"), total_KE_ratios)
    else:
        np.save(os.path.join(save_dir, "total_KE_ratios.npy"), total_KE_ratios)
        complete_cpu_time = np.sum(total_time)
        print(f"total CPUhrs = {complete_cpu_time}")
        np.save(os.path.join(save_dir,"final_time_count.npy"), complete_cpu_time)

    # saving coefficient options if another task hasn't done it yet
    if not os.path.exists(os.path.join(save_dir, "c_nums.npy")):
        np.save(os.path.join(save_dir, "c_nums.npy"),  c_nums)

    ############################################################
    # deleting checkpoints to save space now that all info is combined
    ############################################################
    for k in range(n_ref_steps):
        check_file_path = os.path.join(save_dir, f"check_ref_{k}_task_{task_id}.npy")
        if os.path.exists(check_file_path):
            os.remove(check_file_path)
    
    print(f"Task {task_id} completed and checkpoints cleared")
