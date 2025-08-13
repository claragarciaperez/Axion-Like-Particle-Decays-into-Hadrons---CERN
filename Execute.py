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
Y_SM = 0.2533435042576797 
Neff_SM = 3.0287538204143387
zeta3   = 1.202056903159594  # ζ(3)


#%%
T_0 = 20 #Initial temperature in MeV
T_f = 0.01 #Final temperature in MeV
ma = 400 #ALP mass
Br = defin.Br_pion(ma) #Branching ratio
upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*ma
prop = np.logspace(np.log10(1e-6),np.log10(1), 10) #Values of rhoa/rhoSM
taua = np.logspace(np.log10(0.01),np.log10(20), 10) #Values of taua

Temperature_values = np.logspace(np.log10(T_0), np.log10(T_f), num=1000)

t_list = []
for i in Temperature_values:
    t_list.append(defin.time_RD(i,gsinterp(i)))
time_seconds = []
for t in t_list:
    time_seconds.append(t*h_bar)
#%%
T_0 = 20 #Initial temperature in MeV
T_f = 0.01 #Final temperature in MeV
ma = 900 #ALP mass
Br = defin.Br_pion(ma) #Branching ratio
upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*ma
prop = np.logspace(np.log10(1e-6),np.log10(upper_bound_prop), 20) #Values of rhoa/rhoSM
taua = np.logspace(np.log10(0.01),np.log10(20), 20) #Values of taua

Temperature_values = np.logspace(np.log10(T_0), np.log10(T_f), num=1000)

t_list = []
for i in Temperature_values:
    t_list.append(defin.time_RD(i,gsinterp(i)))
time_seconds = []
for t in t_list:
    time_seconds.append(t*h_bar)

#Neff = []
Y_api = []
#Y_a = []
for p in prop:
    #auxN = []
    auxYapi = []
    #auxYa = []
    for t in taua:
        Heapi= Xn.He_neff(T_0,T_f,t_list,time_seconds,p,t,ma,Br, neff=False,Ya= False)[0]
        #auxN.append(neff)
        auxYapi.append((Heapi- Y_SM)/Y_SM)
        #auxYa.append((Hea- Y_SM)/Y_SM)
    #Neff.append(auxN)
    Y_api.append(auxYapi)
    #Y_a.append(auxYa)

# %%
#defin.paint_Neff(Neff,taua,prop,ma,Br)
defin.paint_Xn_pions(Y_api,taua,prop,ma,Br)
#defin.paint_Xn_no_pions(Y_a,taua,prop,ma,Br)
defin.save_Yapi(Y_api,ma,prop, taua)


