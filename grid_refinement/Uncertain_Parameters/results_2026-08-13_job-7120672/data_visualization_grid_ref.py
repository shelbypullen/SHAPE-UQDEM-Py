import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors

script_dir = os.path.dirname(__file__)                  # get the current file path to grab all data here
figure_dir = os.path.join(script_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)                  # make a new folder to store figures

#load data
c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
total_KE_avgs = np.load(os.path.join(script_dir, "total_KE_avgs.npy"),allow_pickle=True)
total_KE_stds = np.load(os.path.join(script_dir, "total_KE_stds.npy"),allow_pickle=True)
total_KE_cost = np.load(os.path.join(script_dir, "total_KE_cost.npy"),allow_pickle=True)
total_KE_ratios = np.load(os.path.join(script_dir, "total_KE_ratios.npy"),allow_pickle=True)

# confirm that all arrays are the same size
print(c_nums)
print(np.size(total_KE_avgs))
print(np.size(total_KE_stds))
print(np.size(total_KE_cost))
print(np.size(total_KE_ratios))

print(np.size(total_KE_ratios[0]))

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
fig2, axs2 = plt.subplots(1,1,figsize=(8, 8))
norm2 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2.pcolormesh(c3s,c2s, total_KE_avgs[-1],cmap='RdBu_r',norm=norm2)
axs2.set_title(f"{int(c_nums[-1])}x{int(c_nums[-1])}")
axs2.set_xlabel("C3", fontsize=8)
axs2.set_ylabel("c2", fontsize=8)
fig2.colorbar(pcm2, ax=axs2).set_label("KE Ratio", fontsize=6)

axs2.set_xlim([1,15])
axs2.set_ylim([-10,-1])

plt.tight_layout()

# save the plot
file_path_fig2 = os.path.join(figure_dir, f"most_refined_{c_nums[-1]}x{c_nums[-1]}_KE_avgs.png")
fig2.savefig(file_path_fig2, dpi=300, bbox_inches='tight')

# to see the plot
plt.show()