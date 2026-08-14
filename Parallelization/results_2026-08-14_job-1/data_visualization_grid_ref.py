import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors

"""
COPY THIS SCRIPT INTO THE RESULTS FOLDER OR ADD ITS FILE PATH TO THE script_dir
"""

script_dir = os.path.dirname(__file__)                  # get the current file path to grab all data here
figure_dir = os.path.join(script_dir, "Figures")
os.makedirs(figure_dir, exist_ok=True)

#load data
#c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
c_nums = [20]
total_KE_avgs = np.load(os.path.join(script_dir, "KE_avgs.npy"),allow_pickle=True)
total_KE_stds = np.load(os.path.join(script_dir, "KE_stds.npy"),allow_pickle=True)
total_KE_cost = np.load(os.path.join(script_dir, "KE_cost.npy"),allow_pickle=True)
total_KE_ratios = np.load(os.path.join(script_dir, "KE_ratios.npy"),allow_pickle=True)
print(np.shape(total_KE_avgs[0]))
total_KE_avgs[-1] = total_KE_avgs[-1].reshape(20,20)
# confirm that all arrays are the same size
print(c_nums)
print(np.size(total_KE_avgs))
print(np.size(total_KE_stds))
print(np.size(total_KE_cost))
print(np.size(total_KE_ratios))

print(np.size(total_KE_ratios[0]))

# the grid size should be as close to square as possible so round up the sqrt(length(c_nums))
grid_size = int(np.ceil(np.sqrt(int(np.size(c_nums)))))
print('plot grid size = ', grid_size)

if len(c_nums) >1:
    # if statement to get rid of extra rows in subplot grid if necessary
    if grid_size**2 - grid_size > len(c_nums):
        fig1, axs1 = plt.subplots(grid_size-1,grid_size, figsize=(4*(grid_size), 3*(grid_size-1)))
    else:
        fig1, axs1 = plt.subplots(grid_size,grid_size, figsize=(3*grid_size, 3*grid_size))
    axs1 = axs1.flatten()

    fig1.suptitle("KE Ratio over Possible Coefficient Domain for grid size")

    # plot pcolor at each grid resolution
    for i in range(len(axs1)):

        # hid all plots that don't have a resolution level to fill it
        if i >= len(c_nums):
            axs1[i].set_visible(False)
            continue

        # get c2 and c3 arrays because I didn't save them - CHANGE BOUNDS TO MATCH CODE
        c2 = np.linspace(-10,-1,c_nums[i])
        c3 = np.linspace(1,15,c_nums[i])

        # center the color bar at 1
        norm = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[i]), vcenter=1, vmax = np.nanmax(total_KE_avgs[i]))

        # plot the graph
        pcm1 = axs1[i].pcolormesh(c2,c3, total_KE_avgs[i],cmap='RdBu_r',norm=norm)
        axs1[i].set_title(f"{int(c_nums[i])}x{int(c_nums[i])}")
        axs1[i].set_xlabel("C3", fontsize=8)
        axs1[i].set_ylabel("c2", fontsize=8)
        fig1.colorbar(pcm1, ax=axs1[i]).set_label("KE Ratio", fontsize=6)

    # save the figure
    plt.tight_layout()
    file_path_fig1 = os.path.join(figure_dir, f"KE_avgs_grid_ref_{len(c_nums)}_steps.png")
    fig1.savefig(file_path_fig1, dpi=300, bbox_inches='tight')

# plot the highest resolution 
# get c2 and c3 arrays because I didn't save them - CHANGE BOUNDS TO MATCH CODE
c2 = np.linspace(-10,-1,c_nums[-1])
c3 = np.linspace(1,15,c_nums[-1])
print(c2)
fig2, axs2 = plt.subplots(1,1,figsize=(2*(grid_size), 2*(grid_size)))
norm2 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2.pcolormesh(c2,c3, total_KE_avgs[-1],cmap='RdBu_r',norm=norm2)
axs2.set_title(f"{int(c_nums[-1])}x{int(c_nums[-1])}")
axs2.set_xlabel("C3", fontsize=8)
axs2.set_ylabel("c2", fontsize=8)
fig2.colorbar(pcm2, ax=axs2).set_label("KE Ratio", fontsize=6)
plt.tight_layout()

# save the plot
file_path_fig2 = os.path.join(figure_dir, f"most_refined_{c_nums[-1]}x{c_nums[-1]}_KE_avgs.png")
fig2.savefig(file_path_fig2, dpi=300, bbox_inches='tight')

# to see the plot
plt.show()

######################################################
# opt value graphs
######################################################
opt_c2s = np.zeros(len(c_nums))
opt_c3s = np.zeros(len(c_nums))

opt_KEs = np.zeros((len(c_nums),3))
opt_KE_avgs = np.zeros((len(c_nums),3))
opt_KE_std = np.zeros((len(c_nums),3))

# grab all optimal values for plotting
for i in range(len(c_nums)):

    c2 = np.linspace(-10,-1,c_nums[i])
    c3 = np.linspace(1,15,c_nums[i])

    # get optimal values
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_cost[i]), total_KE_cost[i].shape)      # where the optimal spring coefficients are
    opt_c2s[i] = c2[c2_idx]
    opt_c3s[i] = c3[c3_idx]
    opt_KEs[i] = [np.nanmin(total_KE_cost[i]), total_KE_avgs[i][c2_idx,c3_idx], total_KE_stds[i][c2_idx,c3_idx]]
    
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_avgs[i]), total_KE_avgs[i].shape)      # where best avg KE ratio is
    opt_KE_avgs[i] = [np.nanmin(total_KE_avgs[i]), c2[c2_idx], c3[c3_idx]]
    
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_stds[i]), total_KE_stds[i].shape)      # where best std of KE ratio is
    opt_KE_std[i] = [np.nanmin(total_KE_stds[i]), c2[c2_idx], c3[c3_idx]]

# plotting optimal spring coefficients for cost function
fig3, ax3 = plt.subplots(1,1,figsize=(2*(grid_size), (grid_size)))
ax3.plot(c_nums, opt_c2s)
ax3.plot(c_nums, opt_c3s)
ax3.legend(labels=["Optimal C2 Values","Optimal C3 Values"])
ax3.set_title("Change in Optimal Coefficients as Resolution Improves")

file_path_fig3 = os.path.join(figure_dir, "optimal_coeff_values_grid_ref.png")
fig3.savefig(file_path_fig3, dpi=300, bbox_inches='tight')

# plotting optimal KE values
fig4, ax4 = plt.subplots(1,1, figsize=(2*(grid_size), grid_size))
ax4.plot(c_nums, opt_KEs[:,0])
ax4.plot(c_nums, opt_KE_avgs[:,0])
ax4.plot(c_nums,opt_KE_std[:,0])
ax4.legend(labels=["Min Cost Value","Min Average KE ratio","Min std of KE ratios"])
ax4.set_title("Optimal Values as Grid resultion Improves")

# saving the plot
file_path_fig4 = os.path.join(figure_dir, "optimal_KE_values_grid_ref.png")
fig4.savefig(file_path_fig4, dpi=300, bbox_inches='tight')

# to see the plot
plt.show()
