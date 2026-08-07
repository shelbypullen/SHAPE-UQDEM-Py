import multiprocessing
multiprocessing.set_start_method("fork")
from multiprocessing import Pool            # parallelizing tool
import numpy as np
import objective                            # importing objective function file
import os                                   # to get number of cpus
import time                                 # to track computational time
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Ellipse

"""
!!!!!!MAKE SURE TO CHANGE COARSENESS FOR LAPTOP RUNS!!!!!!
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
    
    theta = 295*np.pi/180
    a = 3
    b = 7
    c = 6
    d = -4

    if ((c3-c)*np.cos(theta) - (c2-d)*np.sin(theta))**2/(a**2) + ((c3-c)*np.sin(theta) + (c2-d)*np.cos(theta))**2/(b**2) > 1:
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
    # random input space defined - CAN CHANGE n_samples
    ############################################################
    n_samples = 1
    rng = np.random.default_rng()
    M_up = 0.45 + 0.015
    M_low = 0.45 - 0.015
    #M_normal = rng.uniform(M_low, M_up, n_samples)         # varying inputs

    M_normal = [0.05]                                       #single input normalized mass M/M0

    V_up = 0.8+0.3
    V_low = 0.8-0.3
    #V_normal = rng.uniform(V_low, V_up, n_samples)         # varing inputs
    V_normal = [1]                                          # single input normalized velocity V/V0

    Z_up = 0.15+0.01
    Z_low = 0.15-0.01
    #zeta_sweep = rng.uniform(Z_low,Z_up,n_samples)         # varing inputs
    zeta_sweep = [0.01]                                     # single input
    stochastic_info = [M_normal, V_normal, zeta_sweep]

    ############################################################
    # Defining C2 C3 grid coarseness
    ############################################################
    refinement_factor = 1.3333                              # so max c_nums = 500 after 10 steps (i=0:9)
    #coarse = 20
    coarse = 20
    n_steps = 1
    c_nums = [int(np.round(coarse*(refinement_factor**i))) for i in range(n_steps)]  

    ############################################################
    # Array Objects to store results for each coarseness
    ############################################################
    total_KE_cost   = np.empty(n_steps, dtype=object)
    total_KE_avgs   = np.empty(n_steps, dtype=object)
    total_KE_stds   = np.empty(n_steps, dtype=object)
    total_KE_ratios = np.empty(n_steps, dtype=object)
    opt_c2s = np.zeros(n_steps)
    opt_c3s = np.zeros(n_steps)
    opt_KEs = np.zeros((n_steps, 3))                                  # each row = [opt_cost, opt KE_avg, opt_std]
    ############################################################
    # run objective function
    ############################################################
    with Pool(n_cores, initializer=init_worker, initargs=(stochastic_info,)) as pool:                              

        for k in range(len(c_nums)):
            c_num = c_nums[k]
            c2_sweep = np.linspace(-10,-1,c_num)                   # can change ranges
            c3_sweep = np.linspace(1,15,c_num)                     # can change ranges

            ############################################################
            # empty results arrays
            ############################################################
            KE_avgs   = np.zeros((c_num,c_num))
            KE_stds   = np.zeros((c_num,c_num))
            KE_cost   = np.zeros((c_num,c_num))
            KE_ratios = np.zeros((c_num,c_num, n_samples))

            tasks = [                                               # defining what is being parallelized over
                (i, j, c2_sweep[i], c3_sweep[j], n_samples) 
                for i in range(c_num)
                for j in range(c_num)
            ]

            results = pool.starmap(run_obj, tasks)

            for i, j, cost, avg, std, ratio in results:             # getting the results
                KE_cost[i,j]   = cost
                KE_avgs[i,j]   = avg
                KE_stds[i,j]   = std
                KE_ratios[i,j] = ratio

            total_KE_cost[k]   = KE_cost
            total_KE_avgs[k]   = KE_avgs
            total_KE_stds[k]   = KE_stds
            total_KE_ratios[k] = KE_ratios
            ############################################################
            # get optimal values
            ############################################################
            c2_idx,c3_idx = np.unravel_index(np.nanargmin(KE_cost), KE_cost.shape)      # where the optimal spring coefficients are
            opt_c2s[k] = c2_sweep[c2_idx]
            opt_c3s[k] = c3_sweep[c3_idx]
            opt_KEs[k] = [np.nanmin(KE_cost), KE_avgs[c2_idx,c3_idx], KE_stds[c2_idx,c3_idx]]

    toc = time.perf_counter()                                # getting final time
    print(f"runtime = {toc-tic:.3f}")

    ############################################################
    # Saving data
    ############################################################
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # saving the date and time
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # saving all desired outputs
    np.save(os.path.join(save_dir, "total_KE_avgs.npy"),   total_KE_avgs)
    np.save(os.path.join(save_dir, "total_KE_stds.npy"),   total_KE_stds)
    np.save(os.path.join(save_dir, "total_KE_cost.npy"),   total_KE_cost)
    np.save(os.path.join(save_dir, "total_KE_ratios.npy"), total_KE_ratios)

    # saving coefficient options
    np.save(os.path.join(save_dir, "c_nums.npy"),  c_nums)

    # printing all necessary info to the .out file
    print(f"timestamp: {timestamp}")
    print(f"n_cores: {n_cores}")
    print(f"n_samples: {n_samples}")
    for k in range(n_steps):
        print(f"refinement level {k}: c2 = {opt_c2s[k]:.4f}, c3 = {opt_c3s[k]:.4f}, ")
        print(f"KE_cost = {opt_KEs[k,0]:.4f}, KE_avg = {opt_KEs[k,1]:.4f}, KE_std = {opt_KEs[k,2]:.4f}")

############################################################
#
############################################################
fig, ax = plt.subplots(1,1)

norm = colors.TwoSlopeNorm(vmin=KE_avgs.min(), vcenter=1, vmax = KE_avgs.max())
pcm = ax.pcolormesh(c3_sweep, c2_sweep, KE_cost, cmap='RdBu_r',norm=norm)

t = np.linspace(0, 2*np.pi, 1000)

theta = 295*np.pi/180 - np.pi/4
a = 2
b = 7
c = 6
d = -3.25

# parametric ellipse in rotated frame, then transform back to c2/c3 space
c3_ellipse = c + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
c2_ellipse = d + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)

ax.plot(c3_ellipse, c2_ellipse, 'r-', lw=2)

ax.set_title("Average KE Ratio at varying spring polynomial coefficients")
ax.set_xlabel("C3")
ax.set_xlim(1,15)
ax.set_ylabel("C2")
ax.set_ylim(-10,-1)
fig.colorbar(pcm, ax=ax).set_label("Average KE Ratio")

plt.show()