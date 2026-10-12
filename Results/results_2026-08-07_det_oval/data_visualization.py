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
total_KE_ratios = np.load(os.path.join(script_dir, "total_KE_ratios.npy"),allow_pickle=True)

# confirm that all arrays are the same size
print(c_nums)
print(np.size(total_KE_avgs))
print(np.size(total_KE_stds))
print(np.size(total_KE_cost))
print(np.size(total_KE_ratios))

print(np.size(total_KE_ratios[0]))

# creating ellipse to outline boundaries
t = np.linspace(0, 2*np.pi, 1000)
theta = 295*np.pi/180 - np.pi/4
a = 3
b = 7
c = 6
d = -4

# parametric ellipse in rotated frame, then transform back to c2/c3 space
c3_ellipse = c + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
c2_ellipse = d + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
c3_ellipse = c3_ellipse[np.abs(c2_ellipse+7.25).argmin():]
c2_ellipse = c2_ellipse[np.abs(c2_ellipse+7.25).argmin():]
c2_ellipse = c2_ellipse[:np.abs(c3_ellipse-5).argmin()]
c3_ellipse = c3_ellipse[:np.abs(c3_ellipse-5).argmin()]

c2_line = np.linspace(-7.25,-1, 1000)
c3_line = 2/9*c2_line**2

################################################################
# start plotting
################################################################
# the grid size should be as close to square as possible so round up the sqrt(length(c_nums))
grid_size = int(np.ceil(np.sqrt(int(np.size(c_nums)))))
print('plot grid size = ', grid_size)

if len(c_nums) > 1:
    # if statement to get rid of extra rows in subplot grid if necessary
    if grid_size**2 - grid_size >= len(c_nums):
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
        c2s = np.linspace(-10,-1,c_nums[i])
        c3s = np.linspace(1,15,c_nums[i])

        # center the color bar at 1
        norm = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[i]), vcenter=1, vmax = np.nanmax(total_KE_avgs[i]))

        # plot the graph
        pcm1 = axs1[i].pcolormesh(c3s,c2s, total_KE_avgs[i],cmap='RdBu_r',norm=norm)
        axs1[i].set_title(f"{int(c_nums[i])}x{int(c_nums[i])}")
        axs1[i].set_xlabel("C3", fontsize=8)
        axs1[i].set_ylabel("c2", fontsize=8)
        fig1.colorbar(pcm1, ax=axs1[i]).set_label("KE Ratio", fontsize=6)

        # plotting boundaries of computation
        #axs1[i].plot(c3_ellipse, c2_ellipse, color="black", lw=2)
        #axs1[i].plot(c3_line,c2_line, color="black", lw=2)
        axs1[i].set_xlim([1,15])
        axs1[i].set_ylim([-10,-1])

    # save the figure
    plt.tight_layout()
    file_path_fig1 = os.path.join(figure_dir, f"KE_avgs_grid_ref_{len(c_nums)}_steps.png")
    fig1.savefig(file_path_fig1, dpi=300, bbox_inches='tight')

# plot the highest resolution 
fig2, axs2 = plt.subplots(1,1,figsize=(8, 8))
norm2 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2.pcolormesh(c3s,c2s, total_KE_avgs[-1],cmap='RdBu_r',norm=norm2)
axs2.set_title(f"{int(c_nums[-1])}x{int(c_nums[-1])}")
axs2.set_xlabel("C3", fontsize=8)
axs2.set_ylabel("c2", fontsize=8)
fig2.colorbar(pcm2, ax=axs2).set_label("KE Ratio", fontsize=6)

#axs2.plot(c3_ellipse, c2_ellipse, color="black", lw=2)
#axs2.plot(c3_line,c2_line, color="black", lw=2)
axs2.set_xlim([1,15])
axs2.set_ylim([-10,-1])

plt.tight_layout()

# save the plot
file_path_fig2 = os.path.join(figure_dir, f"most_refined_{c_nums[-1]}x{c_nums[-1]}_KE_avgs.png")
fig2.savefig(file_path_fig2, dpi=300, bbox_inches='tight')

# figure zoomed in
# plot the highest resolution 
fig3, axs3 = plt.subplots(1,1,figsize=(3.5, 3), layout="constrained")

full_cmap = plt.colormaps['RdBu_r']
half_cmap = colors.ListedColormap(full_cmap(np.linspace(0, 0.5, 256)))

norm3 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=0.5, vmax = 1)
pcm3 = axs3.pcolormesh(c3s,c2s, total_KE_avgs[-1], cmap="RdBu_r")#,cmap=half_cmap,norm=norm3)
#cbar3 = fig3.colorbar(pcm3, ax=axs3)

axs3.set_xlim([6,10])
axs3.set_ylim([-6.25,-5])
axs3.set_xlabel("$c_3$", fontsize=12)
axs3.set_ylabel("$c_2$",fontsize=12)

# save the plot
file_path_fig3 = os.path.join(figure_dir, f"zoomed_{c_nums[-1]}x{c_nums[-1]}_KE_avgs.png")
fig3.savefig(file_path_fig3, dpi=300, bbox_inches='tight')


# figure zoomed in
# plot the highest resolution 

fig4, axs4 = plt.subplots(1,1,figsize=(3.5, 3), layout="constrained")

full_cmap = plt.colormaps['RdBu_r']
half_cmap = colors.ListedColormap(full_cmap(np.linspace(0, 0.5, 256)))

norm4 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=0.5, vmax = 1)
pcm4 = axs4.pcolormesh(c3s,c2s, total_KE_avgs[-1], cmap="RdBu_r", norm=norm2)#,cmap=half_cmap,norm=norm3)
#cbar3 = fig3.colorbar(pcm3, ax=axs3)

axs4.set_ylim([-4,-2])
axs4.set_xlim([1,4])
axs4.set_xlabel("$c_3$", fontsize=12)
axs4.set_ylabel("$c_2$",fontsize=12)

#plt.tight_layout()

# save the plot
file_path_fig4 = os.path.join(figure_dir, f"zoomed_top_KE_avgs.png")
fig4.savefig(file_path_fig4, dpi=300, bbox_inches='tight')

# to see the plot
plt.show()