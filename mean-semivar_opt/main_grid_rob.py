import numpy as np
import objective 
from matplotlib import pyplot as plt
from matplotlib import colors
from multiprocessing import Pool
import time

tic = time.perf_counter()                               # counts seconds
############################################################
# Defining C2 C3 space - CAN CHANGE c2_num, c3_num FOR HIGHER PRECISION
############################################################
c2_num = 10                                 
c3_num = 10
c2_sweep = np.linspace(-10,-1,c2_num)                   # can change ranges
c3_sweep = np.linspace(0,15,c3_num)                     # can change ranges


############################################################
# random input space defined - CAN CHANGE n_samples
############################################################
n_samples = 1
rng = np.random.default_rng()
M_up = 0.45 + 0.015
M_low = 0.45 - 0.015
#M_normal = rng.uniform(M_low, M_up, n_samples)         # varying inputs
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
for i in range(c3_num):
    c3 = c3_sweep[i]
    for j in range(c2_num):
        c2 = c2_sweep[j]

        if c3 < 2/9*c2**2:
            KE_avgs[i,j] = np.nan
            KE_stds[i,j] = np.nan
            KE_cost[i,j] = np.nan
            continue
        
        non_spring_info = [c2,c3]
        [KE_cost[i,j], KE_avgs[i,j], KE_stds[i,j], KE_ratios[i,j]] = objective.objective(non_spring_info, stochastic_info, n_samples)

############################################################
# get optimal values
############################################################
c2_idx,c3_idx = np.unravel_index(np.nanargmin(KE_cost), KE_cost.shape)
print(f"c2 = {c2_sweep[c2_idx]}, c3 = {c3_sweep[c3_idx]}")
print(f"Average KE Ratio = {np.nanmin(KE_avgs)}")

toc = time.perf_counter()                           # getting final time
print(f"time = {toc-tic:.3f}")

############################################################
#Plotting
############################################################
norm = colors.TwoSlopeNorm(vmin=KE_avgs.min(), vcenter=1, vmax = KE_avgs.max())
plt.pcolormesh(c3_sweep, c2_sweep, KE_cost, cmap='RdBu_r',norm=norm)
plt.title("Average KE Ratio at varying spring polynomial coefficients")
plt.xlabel("C3")
plt.ylabel("C2")
plt.colorbar().set_label("Average KE Ratio")
plt.show()