import numpy as np

def estimate(n_cores, n_nodes, grid_length_start, ref_factor, n_grids, n_MCs, safety_factor):
    prev_n_cores = 112
    prev_n_nodes = 1
    prev_grid_length = 20
    prev_ref_factor = 1.38
    prev_n_grids = 10
    prev_n_MCs = 1
    prev_time = 1813/3600        #[s]

    i = np.arange(0, prev_n_grids)
    prev_total_calls = np.sum((prev_grid_length*prev_ref_factor**i)**2) * prev_n_MCs

    prev_CPUhr_per_call = prev_time * (prev_n_cores * prev_n_nodes) / prev_total_calls

    j = np.arange(0,n_grids)
    total_calls = np.sum((grid_length_start*(ref_factor**j))**2) * n_MCs
    print(f"Total Function Calls = {total_calls}")

    new_total_hr = prev_CPUhr_per_call * total_calls / (n_cores * n_nodes)
<<<<<<< HEAD

    print(f"Average Estimated Hours to finish = {new_total_hr:.2f}")
    print(f"If on Multiple Nodes, Time for Last Node = {new_total_hr*2:.2f}")

    print(f"Recommended Time Limit = {new_total_hr*safety_factor:.2f} or {new_total_hr*2*safety_factor:.2f}")

# new run info 
n_cores = 112                       # per node
n_nodes = 15
grid_length_start = 200             # like c_num - number in linspace discretizing coeff array

=======
    print(f"Estimated Hours to finish = {new_total_hr:.2f}")

    print(f"Recommended Time Limit = {new_total_hr*safety_factor:.2f}")

# new run info 
n_cores = 112                       # per node
n_nodes = 1
grid_length_start = 20             # like c_num - number in linspace discretizing coeff array
>>>>>>> origin/main
ref_factor = 1.38                   # multiplication factor that the grid length increases by each time
n_grids = 1                         # number of different grids - set as one if no grid refinement
n_MCs = 1000                           # number of Monte carlo samples 
safety_factor = 1.2                 # can change depending on how sure you are about the function call time

# call function - it will print out times
estimate(n_cores, n_nodes, grid_length_start, ref_factor, n_grids, n_MCs, safety_factor)