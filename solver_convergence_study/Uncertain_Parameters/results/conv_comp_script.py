import numpy as np
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(__file__)

c_nums = np.load(os.path.join(script_dir, "c_nums.npy"))
print(int(c_nums[0]))
c2s = np.linspace(-10,-1,int(c_nums[0]))
c3s = np.linspace(1,15,int(c_nums[0]))

IRK5 = np.load(os.path.join(script_dir, "IRK5-total_KE_avgs.npy"), allow_pickle=True)
RK4 = np.load(os.path.join(script_dir, "RK4-total_KE_avgs.npy"),allow_pickle=True)
RK8 = np.load(os.path.join(script_dir, "RK8-total_KE_avgs.npy"),allow_pickle=True)
print(np.shape(IRK5[0]))
I_4 = abs(IRK5[0] - RK4[0])
I_8 = abs(IRK5[0] - RK8[0])
RK8_4 = abs(RK8[0] - RK4[0])

vmin = min(abs(np.nanmin(I_4)),np.nanmin(I_8),np.nanmin(RK8_4))
vmax = max(abs(np.nanmax(I_4)),np.nanmax(I_8),np.nanmax(RK8_4))
print(I_4[-1,-1])
fig, axs = plt.subplots(1,3,figsize=(12,4))

pcm1 = axs[0].pcolormesh(c2s,c3s,I_4, cmap="inferno", vmin=vmin, vmax=vmax)
axs[0].set_title("Implicit RK5 Minus RK45")

pcm2 = axs[1].pcolormesh(c2s,c3s,RK8_4, cmap="inferno", vmin=vmin, vmax=vmax)
axs[1].set_title("RK4 Minus RK45")

pcm3 = axs[2].pcolormesh(c2s,c3s,I_8, cmap="inferno", vmin=vmin, vmax=vmax)
axs[2].set_title("Implicit RK5 Minus RK8")

plt.colorbar(pcm1, ax=axs, location="right")

#plt.tight_layout()
plt.show()