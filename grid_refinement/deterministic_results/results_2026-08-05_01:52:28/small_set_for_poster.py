import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

"""
COPY THIS SCRIPT INTO THE RESULTS FOLDER OR ADD ITS FILE PATH TO THE script_dir
"""
script_dir = os.path.dirname(__file__)                  # pwd to grab all data in the same folder
figure_dir = os.path.join(script_dir, "Figures_poster")
os.makedirs(figure_dir, exist_ok=True)

# load most of the data
c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
total_KE_avgs = np.load(os.path.join(script_dir, "total_KE_avgs.npy"),allow_pickle=True)
total_KE_stds = np.load(os.path.join(script_dir, "total_KE_stds.npy"),allow_pickle=True)
total_KE_cost = np.load(os.path.join(script_dir, "total_KE_cost.npy"),allow_pickle=True)
total_KE_ratios = np.load(os.path.join(script_dir, "total_KE_ratios.npy"),allow_pickle=True)

# load the 500x500 data
script_dir500 = os.path.join(script_dir, "results_2026-08-06_14:45:12")
KE_avgs = np.load(os.path.join(script_dir500, "KE_avgs.npy"),allow_pickle=True)
KE_avgs500 = np.empty(1, dtype=object)
KE_avgs500[0] = KE_avgs
KE_stds = np.load(os.path.join(script_dir500, "KE_stds.npy"),allow_pickle=True)
KE_stds500 = np.empty(1, dtype=object)
KE_stds500[0] = KE_stds
KE_cost = np.load(os.path.join(script_dir500, "KE_cost.npy"),allow_pickle=True)
KE_cost500 = np.empty(1, dtype=object)
KE_cost500[0] = KE_cost
KE_ratios = np.load(os.path.join(script_dir500, "KE_ratios.npy"),allow_pickle=True)
KE_ratios500 = np.empty(1, dtype=object)
KE_ratios500[0] = KE_ratios

# merge data
c_nums = np.append(c_nums, int(500))
total_KE_avgs = np.append(total_KE_avgs, KE_avgs500)
total_KE_stds = np.append(total_KE_stds, KE_stds500)
total_KE_cost = np.append(total_KE_cost, KE_cost500)
total_KE_ratios = np.append(total_KE_ratios, KE_ratios500)


# confirm that all arrays are the same size
print(c_nums)
print(np.size(total_KE_avgs))
print(np.size(total_KE_stds))
print(np.size(total_KE_cost))
print(np.size(total_KE_ratios))

print(np.size(total_KE_ratios[0]))

######################################################
# pcolor mesh of c2 c3 domain
######################################################
# the grid size should be as close to square as possible so round up the sqrt(length(c_nums))
grid_size = 3
print('plot grid size = ', grid_size)


fig1, axs1 = plt.subplots(1,grid_size, figsize=(12, 3.8), layout="constrained")

fig1.suptitle("KE Ratio over Possible Coefficient Domain for grid size", fontsize=15)

# plot pcolor at each grid resolution
wanted = [0,5,10]
i=0
for j in wanted:

    # get c2 and c3 arrays because I didn't save them - CHANGE BOUNDS TO MATCH CODE
    c2 = np.linspace(-10,-1,c_nums[j])
    c3 = np.linspace(1,15,c_nums[j])

    # center the color bar at 1
    norm = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[j]), vcenter=1, vmax = np.nanmax(total_KE_avgs[j]))

    # plot the graph
    pcm1 = axs1[i].pcolormesh(c3,c2, total_KE_avgs[j],cmap='RdBu_r',norm=norm)
    axs1[i].set_title(f"{int(c_nums[j])}x{int(c_nums[j])}", fontsize=12)
    axs1[i].set_xlabel("$c_3$", fontsize=12)
    axs1[i].set_ylabel("$c_2$", fontsize=12)
    axs1[i].set_ylim((-8,-1))

    # red circle around optimal solution
    c2_idx,c3_idx = np.unravel_index(np.nanargmin(total_KE_cost[j]), total_KE_cost[j].shape)
    c2_opt = c2[c2_idx]
    c3_opt = c3[c3_idx]
    circle = plt.Circle((c3_opt,c2_opt), radius=0.5, color='red', fill=False, linewidth=2)
    axs1[i].add_patch(circle)
    i+=1
cbar1 = fig1.colorbar(pcm1, ax=axs1, location="right", pad=0.01)
cbar1.set_label("KE Ratio", fontsize=12)
file_path_fig1 = os.path.join(figure_dir, "grid_ref_comp.png")
fig1.savefig(file_path_fig1, dpi=300, bbox_inches='tight')

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
fig3, ax3 = plt.subplots(1,1,figsize=(6, 2.5),layout="constrained")
c2_line = ax3.plot(c_nums**2, opt_c2s,color="blue", label="Optimal C2 Values",
                   linestyle="dashed", marker="o")
c3_line = ax3.plot(c_nums**2, opt_c3s, color="blue", label="Optimal C3 Values",
                   linestyle="dashed", marker="s")
ax3.set_title("Change in Optimal Vlaues as Resolution Improves")
ax3.set_xlabel('Total Grid Points')
ax3.set_ylabel("Coefficient Values")
ax3.set_xscale("log")
ax3.tick_params(axis='y', labelcolor="blue")

ax4 = ax3.twinx()
KE_line = ax4.plot(c_nums**2, opt_KEs[:,0], color="red", label="$KE_{ratios}$",
                   linestyle="solid", linewidth=2, marker="*")
ax4.tick_params(axis='y',labelcolor='red')
ax4.set_ylabel("$KE_{ratio}$")

handles3, labels3 = ax3.get_legend_handles_labels()
handles4, labels4 = ax4.get_legend_handles_labels()

ax3.legend(handles3 + handles4, labels3 + labels4, loc="center right",fontsize=10)
plt.tight_layout()

file_path_fig3 = os.path.join(figure_dir, "optimal_coeff_values_grid_ref.png")
fig3.savefig(file_path_fig3, dpi=300, bbox_inches='tight')

# # plotting optimal KE values
# fig4, ax4 = plt.subplots(1,1, figsize=(6, 3), layout="constrained")
# ax4.plot(c_nums**2, opt_KEs[:,0])
# ax4.set_title("Optimal KE Ratio as Grid Resolution Improves")
# ax4.set_xlabel("Total Grid Points")
# ax4.set_xscale("log")
# ax4.set_ylabel("Optimal KE Ratio")
# plt.tight_layout()

# # saving the plot
# file_path_fig4 = os.path.join(figure_dir, "optimal_KE_values_grid_ref.png")
# fig4.savefig(file_path_fig4, dpi=300, bbox_inches='tight')

# to see the plot
plt.tight_layout()
plt.show()
