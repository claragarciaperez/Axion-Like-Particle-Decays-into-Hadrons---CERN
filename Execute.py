#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import sys
import os
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final_file")
if module_path not in sys.path:
    sys.path.append(module_path)
import module_definitions as defin
import module_Tnu as TNUEM
import module_Xn as Xn

#%% Load the data for g*
data_gs = np.loadtxt(rf'g_star_T_neutrino_1000_T0_25_Tf_0.01.txt', skiprows=1)
gstar = data_gs[:,2]
T_data = data_gs[:,0]

gsinterp= interp1d(T_data, gstar, kind='cubic', fill_value="extrapolate")

h_bar = 6.582e-22 #MeV

#%%
prop = np.logspace(np.log10(1e-2),np.log10(1), 1) #Values of rhoa/rhoSM
taua = np.logspace(np.log10(0.03),np.log10(10), 1) #Values of taua
T_0 = 20 #Initial temperature in MeV
T_f = 0.01 #Final temperature in MeV
ma = 400 #ALP mass
Br = 0.3 #Branching ratio

Temperature_values = np.logspace(np.log10(T_0), np.log10(T_f), num=1000)

t_list = []
for i in Temperature_values:
    t_list.append(defin.time_RD(i,gsinterp(i)))
time_seconds = []
for t in t_list:
    time_seconds.append(t*h_bar)

Neff = []
Y = []
for p in prop:
    auxN = []
    auxY = []
    for t in taua:
        He, neff = Xn.He_neff(T_0,t_list,time_seconds,p,t,ma,Br)
        auxN.append(neff)
        auxY.append(He)
        print(p,t,He, neff)
    Neff.append(auxN)
    Y.append(auxY)
# %%
