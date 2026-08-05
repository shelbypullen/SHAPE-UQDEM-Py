import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors

script_dir = os.path.dirname(__file__)

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


grid_size = int(np.ceil(np.sqrt(int(np.size(c_nums)))))
print('plot grid size = ', grid_size)
if grid_size**2 - grid_size > len(c_nums):
    fig1, axs1 = plt.subplots(grid_size-1,grid_size, figsize=(4*(grid_size), 3*(grid_size-1)))
else:
    fig1, axs1 = plt.subplots(grid_size,grid_size, figsize=(3*grid_size, 3*grid_size))
axs1 = axs1.flatten()

fig1.suptitle("KE Ratio over Possible Coefficient Domain for grid size")
#fig1.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(len(axs1)):
#for i in range(2):
    if i >= len(c_nums):
        axs1[i].set_visible(False)
        continue

    c2 = np.linspace(-10,-1,c_nums[i])
    c3 = np.linspace(1,15,c_nums[i])

    norm = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[i]), vcenter=1, vmax = np.nanmax(total_KE_avgs[i]))

    pcm1 = axs1[i].pcolormesh(c2,c3, total_KE_avgs[i],cmap='RdBu_r',norm=norm)
    axs1[i].set_title(f"{int(c_nums[i])}x{int(c_nums[i])}")
    axs1[i].set_xlabel("C3", fontsize=8)
    axs1[i].set_ylabel("c2", fontsize=8)
    fig1.colorbar(pcm1, ax=axs1[i]).set_label("KE Ratio", fontsize=6)


#fig1.savefig(f"KE_avgs_grid_ref_{len(c_nums)}_steps")
#plt.tight_layout()
#plt.show()

fig2, axs2 = plt.subplots(1,1,figsize=(2*(grid_size), 2*(grid_size)))
norm2 = colors.TwoSlopeNorm(vmin=np.nanmin(total_KE_avgs[-1]), vcenter=1, vmax = np.nanmax(total_KE_avgs[-1]))
pcm2 = axs2.pcolormesh(c2,c3, total_KE_avgs[-1],cmap='RdBu_r',norm=norm2)
axs2.set_title(f"{int(c_nums[-1])}x{int(c_nums[-1])}")
axs2.set_xlabel("C3", fontsize=8)
axs2.set_ylabel("c2", fontsize=8)
fig2.colorbar(pcm2, ax=axs2).set_label("KE Ratio", fontsize=6)

plt.tight_layout()
plt.show()