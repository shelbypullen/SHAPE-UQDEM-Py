import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

script_dir = os.path.dirname(__file__)                  # get the current file path to grab all data here
figure_dir = os.path.join(script_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)                  # make a new folder to store figures

#load data
c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
total_KE_avgs = np.load(os.path.join(script_dir, "total_KE_avgs.npy"),allow_pickle=True)
total_KE_stds = np.load(os.path.join(script_dir, "total_KE_stds.npy"),allow_pickle=True)
total_KE_cost = np.load(os.path.join(script_dir, "total_KE_cost.npy"),allow_pickle=True)
#total_KE_ratios = np.load(os.path.join(script_dir, "total_KE_ratios.npy"),allow_pickle=True)

# confirm that all arrays are the same size
print(c_nums)
print(np.size(total_KE_avgs))
print(np.size(total_KE_stds))
print(np.size(total_KE_cost))
#print(np.size(total_KE_ratios))

################################################################
# start plotting
################################################################

c2s = np.linspace(-10,-1,c_nums[0])
c3s = np.linspace(1,15,c_nums[0])

# plot the highest resolution 
fig2, axs2 = plt.subplots(1,3,figsize=(12, 3.8), layout='constrained')
fig2.suptitle("MC Estimate of 1000 Realizations of $\mathbf{Z}$")
norm2 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2[0].pcolormesh(c3s,c2s, total_KE_avgs[-1],cmap='RdBu_r',norm=norm2)
axs2[0].set_title("Expectation of KE ratio")
axs2[0].set_xlabel("$c_3$", fontsize=12)
axs2[0].set_ylabel("$c_2$", fontsize=12)
cbar2 = fig2.colorbar(pcm2, ax=axs2[0])
cbar2.set_ticks([1])

# red circle around optimal solution
c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_avgs[-1]), total_KE_avgs[-1].shape)
c2_opt = c2s[c2_idx]
c3_opt = c3s[c3_idx]
print(f"best avg = {np.nanmin(total_KE_avgs[-1])} at c2 = {c2_opt}, c3 = {c3_opt}")
circle0 = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
axs2[0].add_patch(circle0)

axs2[0].set_xlim([1,15])
axs2[0].set_ylim([-8,-1])

norm3 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_stds[-1]), vcenter=1, vmax = np.nanmax(total_KE_stds[-1]))
pcm3 = axs2[1].pcolormesh(c3s,c2s, total_KE_stds[-1],cmap='RdBu_r',norm=norm3)
axs2[1].set_title("SD of KE Ratio")
axs2[1].set_xlabel("$c_3$", fontsize=12)
axs2[1].set_ylabel("$c_2$", fontsize=12)
cbar3=fig2.colorbar(pcm3, ax=axs2[1])
cbar3.set_ticks([1])

# red circle around optimal solution
c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_stds[-1]), total_KE_stds[-1].shape)
c2_opt = c2s[c2_idx]
c3_opt = c3s[c3_idx]
print(f"best std = {np.nanmin(total_KE_stds[-1])} at c2 = {c2_opt}, c3 = {c3_opt}")
circle1 = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
axs2[1].add_patch(circle1)

axs2[1].set_xlim([1,15])
axs2[1].set_ylim([-8,-1])

norm4 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_cost[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2[2].pcolormesh(c3s,c2s, total_KE_cost[-1],cmap='RdBu_r',norm=norm4)
axs2[2].set_title("$J(c_2,c_3)$")
axs2[2].set_xlabel("$c_3$", fontsize=12)
axs2[2].set_ylabel("$c_2$", fontsize=12)
cbar4=fig2.colorbar(pcm2, ax=axs2[2])
cbar4.set_ticks([1])

# red circle around optimal solution
c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_cost[-1]), total_KE_cost[-1].shape)
c2_opt = c2s[c2_idx]
c3_opt = c3s[c3_idx]
print(f"best cost = {np.nanmin(total_KE_cost[-1])} at c2 = {c2_opt}, c3 = {c3_opt}")
circle2 = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
axs2[2].add_patch(circle2)

axs2[2].set_xlim([1,15])
axs2[2].set_ylim([-8,-1])

#plt.tight_layout()

# save the plot
file_path_fig2 = os.path.join(figure_dir, f"most_refined_{c_nums[-1]}x{c_nums[-1]}_KE_avgs.png")
fig2.savefig(file_path_fig2, dpi=300, bbox_inches='tight')

# to see the plot
plt.show()