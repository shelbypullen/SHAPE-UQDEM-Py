import numpy as np
import objective 
from matplotlib import pyplot as plt
import time

tic = time.perf_counter()                               # counts seconds
############################################################
# Defining C2 C3 space - CAN CHANGE c2_num, c3_num FOR HIGHER PRECISION
############################################################
c2 = -4.5
c3 = 5
non_spring_info = [c2,c3]

############################################################
# random input space defined - CAN CHANGE n_samples
############################################################
n_samples = 1
M_normal = [0.05]                                       #single input
V_normal = [1]                                          # single input
zeta_sweep = [0.01]                                     # signle input
stochastic_info = [M_normal, V_normal, zeta_sweep]

# changing tolerance terms
tol_num = 28
rel_tol_sweep = np.logspace(-4,-10,tol_num)

KE_cost = np.zeros(tol_num)
KE_avgs = np.zeros(tol_num)
KE_stds = np.zeros(tol_num)
KE_ratios = np.zeros((tol_num, n_samples))

############################################################
# run objective function
############################################################

for k in range(tol_num):
    rel_tol = rel_tol_sweep[k]
    [KE_cost[k], KE_avgs[k], KE_stds[k], KE_ratios[k]] = objective.objective(non_spring_info, stochastic_info, n_samples, rel_tol)

############################################################
# get optimal values
############################################################
KE_avg_dif = np.zeros((tol_num -1,3))
for i in range(tol_num-1):
    KE_avg_dif[i] = [rel_tol_sweep[i], KE_cost[i] - KE_cost[-1], KE_avgs[i] - KE_avgs[-1]]

print(KE_avg_dif)

toc = time.perf_counter()                           # getting final time
print(f"time = {toc-tic:.3f}")

############################################################
# Plotting Graph
############################################################
plt.plot(rel_tol_sweep, KE_avgs)
plt.title("Convergence of KE Ratio as ODE Relative Tolerance Decreases")
plt.xlabel("Tolerance")
plt.ylabel("KE Ratio")
plt.xscale("log")
plt.show()