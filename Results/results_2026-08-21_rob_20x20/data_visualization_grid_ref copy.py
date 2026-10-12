import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter

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

#print(np.size(total_KE_ratios[0]))

# empty arrays to put c2 and c3 arrays for dif resolutions
c2s = np.empty(len(c_nums))
c3s = np.empty(len(c_nums))

################################################################
# start plotting
################################################################
# the grid size should be as close to square as possible so round up the sqrt(length(c_nums))
grid_size = int(np.ceil(np.sqrt(int(np.size(c_nums)))))
print('plot grid size = ', grid_size)

if len(c_nums) > 1:
    # if statement to get rid of extra rows in subplot grid if necessary
    if grid_size**2 - grid_size > len(c_nums):
        fig1, axs1 = plt.subplots(grid_size-1,grid_size, figsize=(4*(grid_size), 3*(grid_size-1)))
    else:
        fig1, axs1 = plt.subplots(grid_size,grid_size, figsize=(3*grid_size, 3*grid_size))

    fig1.suptitle("KE Ratio over Possible Coefficient Domain for grid size")

    # plot pcolor at each grid resolution
    for i in range(len(axs1)):

        # hid all plots that don't have a resolution level to fill it
        if i >= len(c_nums):
            axs1[i].set_visible(False)
            continue

        # get c2 and c3 arrays because I didn't save them - CHANGE BOUNDS TO MATCH CODE
        c2s[i] = np.linspace(-10,-1,c_nums[i])
        c3s[i] = np.linspace(1,15,c_nums[i])

        # center the color bar at 1
        norm = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[i]), vcenter=1, vmax = np.nanmax(total_KE_avgs[i]))

        # plot the graph
        pcm1 = axs1[i].pcolormesh(c2s[i],c3s[i], total_KE_avgs[i],cmap='RdBu_r',norm=norm)
        axs1[i].set_title(f"{int(c_nums[i])}x{int(c_nums[i])}")
        axs1[i].set_xlabel("C3", fontsize=8)
        axs1[i].set_ylabel("c2", fontsize=8)
        fig1.colorbar(pcm1, ax=axs1[i]).set_label("KE Ratio", fontsize=6)

        # plotting boundaries of computation
        axs1[i].set_xlim([1,15])
        axs1[i].set_ylim([-10,-1])

    # save the figure
    plt.tight_layout()
    file_path_fig1 = os.path.join(figure_dir, f"KE_avgs_grid_ref_{len(c_nums)}_steps.png")
    fig1.savefig(file_path_fig1, dpi=300, bbox_inches='tight')

c2s = np.linspace(-10,-1,c_nums[0])
c3s = np.linspace(1,15,c_nums[0])

# plot the highest resolution 
fig2, axs2 = plt.subplots(1,3,figsize=(12, 3.5), layout="constrained")
norm2 = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax = 2)

print(f"avgs = {np.nanmin(total_KE_avgs[-1])}, {np.nanmax(total_KE_avgs[-1])}")
print(f"stds = {np.nanmin(total_KE_stds[-1])}, {np.nanmax(total_KE_stds[-1])}")
print(f"cost = {np.nanmin(total_KE_cost[-1])}, {np.nanmax(total_KE_cost[-1])}")
pcm2 = axs2[0].pcolormesh(c3s,c2s, total_KE_avgs[-1],cmap='RdBu_r', norm=norm2)#,norm=norm2)
axs2[0].set_title("Expectation of KE ratio")
axs2[0].set_xlabel("$c_3$", fontsize=12)
axs2[0].set_ylabel("$c_2$", fontsize=12)
cbar2=fig2.colorbar(pcm2, ax=axs2[0],)
cbar2.set_ticks([0,0.5,1,1.5,2])
cbar2.set_ticklabels(["0","0.5","1","1.5","2"])

# red circle around optimal solution
c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_avgs[-1]), total_KE_avgs[-1].shape)
c2_opt = c2s[c2_idx]
c3_opt = c3s[c3_idx]
print(f"best avg = {np.nanmin(total_KE_avgs[-1])} at c2 = {c2_opt}, c3 = {c3_opt}")
circle0 = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
axs2[0].add_patch(circle0)

axs2[0].set_xlim([1,15])
axs2[0].set_ylim([-8,-1])

norm3 = colors.TwoSlopeNorm(vmin=0, vcenter=0.3, vmax = 0.6)
pcm3 = axs2[1].pcolormesh(c3s,c2s, total_KE_stds[-1],cmap='RdBu_r',norm=norm3)
axs2[1].set_title("Upper Standard Deviation of KE Ratio")
axs2[1].set_xlabel("$c_3$", fontsize=12)
axs2[1].set_ylabel("$c_2$", fontsize=12)
cbar3 = fig2.colorbar(pcm3, ax=axs2[1])
cbar3.set_ticks([0,0.15,0.3,.45,0.6])
cbar3.set_ticklabels(["0","0.15","0.3","0.45","0.6"])


# red circle around optimal solution
c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_stds[-1]), total_KE_stds[-1].shape)
c2_opt = c2s[c2_idx]
c3_opt = c3s[c3_idx]
print(f"best std = {np.nanmin(total_KE_stds[-1])} at c2 = {c2_opt}, c3 = {c3_opt}")
circle1 = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
axs2[1].add_patch(circle1)

axs2[1].set_xlim([1,15])
axs2[1].set_ylim([-8,-1])

norm4 = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax = 2)

pcm2 = axs2[2].pcolormesh(c3s,c2s, total_KE_cost[-1],cmap='RdBu_r',norm=norm4)
axs2[2].set_title("$J_{\mathrm{rob}}(c_2,c_3)$")
axs2[2].set_xlabel("$c_3$", fontsize=12)
axs2[2].set_ylabel("$c_2$", fontsize=12)
cbar4 = fig2.colorbar(pcm2, ax=axs2[2])
cbar4.set_ticks([0,0.5,1,1.5,2])
cbar4.set_ticklabels(["0","0.5","1","1.5","2"])

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