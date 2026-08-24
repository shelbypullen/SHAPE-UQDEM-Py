import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

script_dir = os.path.dirname(__file__)

c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
print(int(c_nums[0]))
c2s = np.linspace(-10,-1,int(c_nums[0]))
c3s = np.linspace(1,15,int(c_nums[0]))

IRK5 = np.load(os.path.join(script_dir, "IRK5-total_KE_avgs.npy"), allow_pickle=True)
RK4 = np.load(os.path.join(script_dir, "RK4-total_KE_avgs.npy"),allow_pickle=True)
RK8 = np.load(os.path.join(script_dir, "RK8-total_KE_avgs.npy"),allow_pickle=True)
RK12 = np.load(os.path.join(script_dir, "RK12-total_KE_avgs.npy"),allow_pickle=True)
print(np.shape(IRK5[0]))
I_4 = abs((IRK5[0] - RK4[0])/RK4[0])*100
I_8 = abs((IRK5[0] - RK8[0])/RK8[0])*100
RK8_4 = abs((RK8[0] - RK4[0])/RK4[0])*100
RK12_4 = abs((RK12[0]-RK4[0]))/RK4[0]*100

# vmin = min(np.nanmin(I_4),np.nanmin(I_8),np.nanmin(RK8_4))
# vmax = max(np.nanmax(I_4),np.nanmax(I_8),np.nanmax(RK8_4))
# norm = LogNorm(vmin,vmax)

# fig, axs = plt.subplots(1,3,figsize=(10,3),dpi=300, constrained_layout=True)

# pcm1 = axs[0].pcolormesh(c3s,c2s,I_4, cmap="inferno", norm=norm)
# axs[0].set_title("Implicit RK5 vs. RK45", fontsize=12)

# pcm2 = axs[1].pcolormesh(c3s,c2s,RK8_4, cmap="inferno", norm=norm)
# axs[1].set_title("RK8 vs. RK45")

# pcm3 = axs[2].pcolormesh(c3s,c2s,I_8, cmap="inferno", norm=norm)
# axs[2].set_title("Implicit RK5 vs. RK8")

# plt.colorbar(pcm1, ax=axs, location="right").set_label("Percent Error [%]")

# fig.savefig("solver_conv_study.png", dpi=300)

#########
norm = LogNorm(np.nanmin(RK12_4),np.nanmax(RK12_4))
fig2, ax = plt.subplots(1,1,figsize=(4,3),dpi=300, constrained_layout=True)

pcm2 = ax.pcolormesh(c3s,c2s,RK12_4, cmap="inferno", norm=norm)
ax.set_title("RK12 vs. RK45", fontsize=15)
ax.set_ylim((-8,-1))
ax.set_ylabel("$c_2$")
ax.set_xlabel("$c_3$")

plt.colorbar(pcm2, ax=ax, location="right").set_label("Percent Error [%]")

fig2.savefig("RK12solver_conv_study.png", dpi=300)

#plt.tight_layout()
plt.show()