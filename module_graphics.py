#%%
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import module_definitions as defin
#%% Constants
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G_F = 1.16638e-11
zeta3   = 1.202056903159594  # ζ(3)
eta0    = 6.1e-10            # baryon-to-photon ratio todat

#%%
def paint_Neff (Neff, taua, prop, mu0 = 2.81, sigma0 = 0.12):
    """
    Color map of Neff. xaxis: taua, yaxis:prop. It differentiates with a dotted red line 
    the accepted and excluded regions. The black area corresponds to the prefered region.

    Parameters
    --------------
        Neff: [float]
            List with the values of Neff. 
        taua: [float]
            List with the values of taua (s)
        prop: [float]
            List with the values of rhoa/rho_SM (0)
        mu0: float
            Measured value of Neff
        sigma0: float
            Error of measured value of Neff
    """
    Neff = np.array(Neff)  
    plt.figure(figsize=(8,6))
    plt.imshow(Neff, 
            extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(prop[0]), np.log10(prop[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')

    cbar = plt.colorbar()
    cbar.set_label(label=r'$N_{\mathrm{eff}}$', 
        fontsize=16,      
        labelpad=15       
    )
    cbar.ax.tick_params(labelsize=12)  

    plt.xlabel(r'$\tau_a (s)$', fontsize = 14)
    plt.ylabel(r'$\rho_a/\rho_{SM}$', fontsize = 14)
    plt.title(rf'$N_{{\mathrm{{eff}}}}$ for $\tau_a$ and $\rho_a/\rho_{{SM}} (T_0)$',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  
    tick_positions = np.log10(tick_vals)

    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    z = (Neff - mu0)/sigma0
    p_two = 2 * stats.norm.sf(np.abs(z))

    X, Y = np.meshgrid(np.log10(taua), np.log10(prop))

    CS = plt.contour(X, Y, p_two, levels = [0.05],colors='red', linestyles='--', linewidths=2)
    plt.contourf(X, Y, Neff, levels=[mu0-sigma0, mu0+sigma0], colors='none',
             hatches=['---'], alpha=0)
    plt.show()
# %%
def paint_Xn_pions (Y_api, taua, prop, ma, Br):
    """
    Color map of (Xn^(a+pi) -Xn_SM)/(Xn^SM). xaxis: taua, yaxis:prop. It differentiates with a dotted red line 
    the accepted and excluded regions. 

    Parameters
    --------------
        Y_api: [float]
            List with the values of (Xn^(a+pi) -Xn_SM)/(Xn^SM). 
        taua: [float]
            List with the values of taua (s)
        prop: [float]
            List with the values of rhoa/rho_SM (0)
        ma: float
            Mass of the ALP in MeV
        Br: float
            Branching ratio of the ALPs in pions
    """
    Y_api = np.array(Y_api) 
    plt.figure(figsize=(8,6))
    plt.imshow(Y_api, 
            extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(prop[0]), np.log10(prop[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\frac{X_n^{(a + \pi)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$', 
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar

    plt.xlabel(r'$\tau_a (s)$', fontsize = 14)
    plt.ylabel(r'$\rho_a/\rho_{SM}$', fontsize = 14)
    plt.title(rf'$\frac{{X_n^{{(a + \pi)}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ for $m_a = {ma}$ MeV and $Br = {Br:.2e}$',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    X = np.log10(taua)
    Y = np.log10(prop)
    CS = plt.contour(X, Y, Y_api, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)
    plt.show()
    plt.show()
# %%
def paint_Xn_no_pions (Y_a, taua, prop):
    """
    Color map of (Xn^(a) -Xn_SM)/(Xn^SM). xaxis: taua, yaxis:prop. It differentiates with a dotted red line 
    the accepted and excluded regions. 

    Parameters
    --------------
        Y_api: [float]
            List with the values of (Xn^(a) -Xn_SM)/(Xn^SM). 
        taua: [float]
            List with the values of taua (s)
        prop: [float]
            List with the values of rhoa/rho_SM (0)
    """
    Y_a = np.array(Y_a)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Y_a, 
            extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(prop[0]), np.log10(prop[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
           interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\frac{X_n^{(a)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$',
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar

    plt.xlabel(r'$\tau_a (s)$', fontsize = 14)
    plt.ylabel(r'$\rho_a/\rho_{SM}$', fontsize = 14)
    plt.title(rf'$\frac{{X_n^{{(a)}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ for $\tau_a$ and $\rho_a/\rho_{{SM}} (T_0)$',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    X = np.log10(taua)
    Y = np.log10(prop)
    CS = plt.contour(X, Y, Y_a, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)
    plt.show()

#%%
def save_Yapi (Yapi, ma, prop, taua):
    """
    Saves the data of Yapi for prop, taua in a file

    Parameters
    --------------
        Yapi: [float]
            List with the values of (Xn^(a+pi) -Xn_SM)/(Xn^SM). 
        ma: float
            Mass of the ALP in MeV
        prop: [float]
            List with the values of rhoa/rho_SM (0)
        taua: [float]
            List with the values of taua (s)
    """
    Y_api_array = np.array(Yapi)

    dimx = len(taua)
    dimy = len(prop)
    # Creamos un archivo de texto
    with open(rf"Final_file\Yapi_ma_{ma}_{dimx}_{dimy}_nobounds.txt", "w") as f:
        # Escribimos la cabecera
        f.write("prop\ttaua\tY_api\n")
        # Iteramos sobre todas las combinaciones
        for i, p in enumerate(prop):
            for j, t in enumerate(taua):
                f.write(f"{p}\t{t}\t{Y_api_array[i,j]}\n")

#%%
def save_Yapi_F (Yapi, ma, F, taua):
    """
    Saves the data of Yapi for F, taua in a file

    Parameters
    --------------
        Yapi: [float]
            List with the values of (Xn^(a+pi) -Xn_SM)/(Xn^SM). 
        ma: float
            Mass of the ALP in MeV
        F: [float]
            List with the values of rhoa/rho_SM (taua)
        taua: [float]
            List with the values of taua (s)
    """
    Y_api_array = np.array(Yapi)

    dimx = len(taua)
    dimy = len(F)
    # Creamos un archivo de texto
    with open(rf"Final_file\Yapi_ma_{ma}_{dimx}_{dimy}_F.txt", "w") as f:
        # Escribimos la cabecera
        f.write("F\ttaua\tY_api\n")
        # Iteramos sobre todas las combinaciones
        for i, p in enumerate(F):
            for j, t in enumerate(taua):
                f.write(f"{p}\t{t}\t{Y_api_array[i,j]}\n")



#%%
def excluded (T_0,prop, taua,gsinterp, mu0,sigma0):
    """
    Graph indicating allowed and forbidden regions in prop, taua. For Neff, Ye with and without pions

    Parameters
    --------------
        T_0: float
            Initial Temperature
        ma: float
            Mass of the ALP in MeV
        prop: [float]
            List with the values of rhoa/rho_SM (0)
        taua: [float]
            List with the values of taua (s)
        gsinterp: interp1d
            Interpolation function to calculate g*(T)
        mu0: float
            Measured value of Neff
        sigma0: float
            Error of measured value of Neff
    """
    data_neff = np.loadtxt('Final_file/Neff_Ya.txt', skiprows=1)
    data_900 = np.loadtxt('Final_file/Yapi_ma_900_20_20.txt', skiprows=1)
    data_600 = np.loadtxt('Final_file/Yapi_ma_600_20_20.txt', skiprows=1)
    data_400 = np.loadtxt('Final_file/Yapi_ma_400_20_20.txt', skiprows=1)
    data_290 = np.loadtxt('Final_file/Yapi_ma_290_20_20.txt', skiprows=1)

    N = data_neff[:,2]
    Ya = data_neff[:,3]
    Yapi_900 = data_900[:,2]
    Yapi_600 = data_600[:,2]
    Yapi_400 = data_400[:,2]
    Yapi_290 = data_290[:,2]

    dat_n, dat_ya = [], []
    dat_yapi_900, dat_yapi_600, dat_yapi_400, dat_yapi_290 = [], [], [], []
    i = 0
    for p in prop:
        aux_n, aux_ya = [], []
        aux_900, aux_600, aux_400, aux_290 = [], [], [], []
        for t in taua:
            aux_n.append(N[i])
            aux_ya.append(Ya[i])
            aux_900.append(Yapi_900[i])
            aux_600.append(Yapi_600[i])
            aux_400.append(Yapi_400[i])
            aux_290.append(Yapi_290[i])
            i += 1
        dat_n.append(aux_n)
        dat_ya.append(aux_ya)
        dat_yapi_900.append(aux_900)
        dat_yapi_600.append(aux_600)
        dat_yapi_400.append(aux_400)
        dat_yapi_290.append(aux_290)

    # Convertir a arrays
    dat_n = np.array(dat_n)
    dat_ya = np.array(dat_ya)
    dat_yapi_900 = np.array(dat_yapi_900)
    dat_yapi_600 = np.array(dat_yapi_600)
    dat_yapi_400 = np.array(dat_yapi_400)
    dat_yapi_290 = np.array(dat_yapi_290)

    plt.figure(figsize=(9,6))



    prop1 = np.logspace(np.log10(1e-6),np.log10(1), 20) #Values of rhoa/rhoSM
    T, P1 = np.meshgrid(taua, prop1)

    z = (dat_n - mu0)/sigma0
    p_two = 2 * stats.norm.sf(np.abs(z))
    # Región roja
    #z_red = np.where(mask_red, 1, 0)
    plt.contour(T, P1, p_two, levels=[0.05], colors='red', linewidths=2, zorder=3)
    plt.contourf(T, P1, p_two, levels=[0,0.05], colors='red', alpha=0.2)
    plt.contourf(T, P1, dat_n, levels=[mu0-sigma0, mu0+sigma0], colors='none',
             hatches=['---'], alpha=0)

    # Región verde
    #mask_Ya = dat_ya>abs(0.01)
    #z_green = np.where(mask_Ya, 1, 0)
    plt.contour(T, P1, dat_ya, levels=[-0.02,0.02], colors='blue', linewidths=2, zorder=5)
    plt.contourf(T, P1, dat_ya, levels=[-np.inf,-0.02], colors='blue', alpha=0.2)
    plt.contourf(T, P1, dat_ya, levels=[0.02,np.inf], colors='blue', alpha=0.2)

    upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*900
    prop3 = np.logspace(np.log10(1e-6),np.log10(upper_bound_prop), 20) #Values of rhoa/rhoSM
    T, P3 = np.meshgrid(taua, prop3)
    # Región azul - Yapi 900
    #mask_yapi_900 = dat_yapi_900 > abs(0.01)
    #z_yapi_900 = np.where(mask_yapi_900, 1, 0)
    plt.contour(T, P3, dat_yapi_900, levels=[-0.02,0.02], colors='green', linewidths=2, zorder=7)

    upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*600
    prop4 = np.logspace(np.log10(1e-6),np.log10(upper_bound_prop), 20) #Values of rhoa/rhoSM
    T, P4 = np.meshgrid(taua, prop4)
    # Región púrpura - Yapi 600
    plt.contour(T, P4, dat_yapi_600, levels=[-0.02,0.02], colors='#FFD700', linewidths=2, zorder=9)

    upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*400
    prop5 = np.logspace(np.log10(1e-6),np.log10(upper_bound_prop), 20) #Values of rhoa/rhoSM
    T, P5 = np.meshgrid(taua, prop5)
    # Región naranja - Yapi 400
    plt.contour(T, P5, dat_yapi_400, levels=[-0.02,0.02], colors='#ff7f00', linewidths=2, zorder=11)

    upper_bound_prop = 2 * zeta3 / np.pi**2 * T_0**3*(gsinterp(T_0)/80)/(defin.rho_SM(T_0,gsinterp(T_0)))*290
    prop6 = np.logspace(np.log10(1e-6),np.log10(upper_bound_prop), 20) #Values of rhoa/rhoSM
    T, P6= np.meshgrid(taua, prop6)
    # Región marrón - Yapi 290
    #mask_yapi_290 = dat_yapi_290 > abs(0.01)
    #z_yapi_290 = np.where(mask_yapi_290, 1, 0)
    plt.contour(T, P6, dat_yapi_290, levels=[-0.02,0.02], colors='#a65628', linewidths=2, zorder=13)


    # Leyenda
    legend_elements = [
        Patch(facecolor='#e41a1c', alpha=1, label=r'$N_{\rm eff}$'),
        Patch(facecolor='blue', alpha=1, label=r'No $\pi$ decay'),
        Patch(facecolor='green', alpha=1, label=r'$m_a = 900$MeV'),
        Patch(facecolor='#FFD700', alpha=1, label=r'$m_a = 600$MeV'),
        Patch(facecolor='#ff7f00', alpha=1, label=r'$m_a = 400$MeV'),
        Patch(facecolor='#a65628', alpha=1, label=r'$m_a = 290$MeV'),
    ]

# Diccionario para iterar sobre cada conjunto
    data_sets = {
        'green': (dat_yapi_900, P3, prop3),
        '#FFD700': (dat_yapi_600, P4, prop4),
        '#ff7f00': (dat_yapi_400, P5, prop5),
        '#a65628': (dat_yapi_290, P6, prop6)
    }

    for color, (dat, P, prop) in data_sets.items():
        y_target = prop.max()
        row_mask = np.isclose(P, y_target)
        x_vals = T[row_mask]
        dat_row = dat[row_mask]

        # Índices donde la fila cruza 0.01
        cross_idx = np.where((dat_row[:-1] <= 0.01) & (dat_row[1:] > 0.01))[0]

        if len(cross_idx) > 0:
            i = cross_idx[0]
            # interpolación lineal para hallar cruce exacto
            x_cross = x_vals[i] + (0.01 - dat_row[i]) * (x_vals[i+1] - x_vals[i]) / (dat_row[i+1] - dat_row[i])

            plt.hlines(y=y_target, xmin=taua.min(), xmax=x_cross,
                    colors=color, linestyles='--', linewidth=2)

    # Escalas y etiquetas
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau_a$ [s]', fontsize=16)
    plt.ylabel(r'$\rho_a/\rho_{SM} (T_0)$', fontsize=16)

    # Leyenda a la derecha
    plt.legend(handles=legend_elements, loc='center left',
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14)

    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 14)
    plt.tight_layout()
    plt.show()
    
# %%
def paint_Xn_pions_F(Y_api, taua, F, ma, Br):
    """
    Color map of (Xn^(a+pi) -Xn_SM)/(Xn^SM). xaxis: taua, yaxis:F. It differentiates with a dotted red line 
    the accepted and excluded regions. 

    Parameters
    --------------
        Y_api: [float]
            List with the values of (Xn^(a+pi) -Xn_SM)/(Xn^SM). 
        taua: [float]
            List with the values of taua (s)
        F: [float]
            List with the values of rhoa/rho_SM (taua)
        ma: float
            Mass of the ALP in MeV
        Br: float
            Branching ratio of the ALPs in pions
    """
    Y_api = np.array(Y_api)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Y_api, 
            extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(F[0]), np.log10(F[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(
        label=r'$\frac{X_n^{(a + \pi)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$', 
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar


    plt.xlabel(r'$\tau_a (s)$', fontsize = 14)
    plt.ylabel(r'$f_{\tau_a}$', fontsize = 14)
    plt.title(
        rf'$\frac{{X_n^{{(a + \pi)}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ for $m_a = {ma}$ MeV and $Br = {Br:.2e}$',
        fontsize=16,
        y=1.05,
        loc='center'
    )

    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    X = np.log10(taua)
    Y = np.log10(F)
    CS = plt.contour(X, Y, Y_api, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)

    plt.show()


# %%
def paint_Xn_no_pions_F(Y_api, taua, F):
    """
    Color map of (Xn^(a) -Xn_SM)/(Xn^SM). xaxis: taua, yaxis:F. It differentiates with a dotted red line 
    the accepted and excluded regions. 

    Parameters
    --------------
        Y_api: [float]
            List with the values of (Xn^(a) -Xn_SM)/(Xn^SM). 
        taua: [float]
            List with the values of taua (s)
        F: [float]
            List with the values of rhoa/rho_SM (taua)

    """
    Y_api = np.array(Y_api)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Y_api, 
            extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(F[0]), np.log10(F[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(
        label=r'$\frac{X_n^{(a)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$', 
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar


    plt.xlabel(r'$\tau_a (s)$', fontsize = 14)
    plt.ylabel(r'$f_{\tau_a}$', fontsize = 14)
    plt.title(
        rf'$\frac{{X_n^{{(a )}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ for different $\tau_a$ and $f_{{\tau_a}}$',
        fontsize=16,
        y=1.05,
        loc='center'
    )

    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    X = np.log10(taua)
    Y = np.log10(F)
    CS = plt.contour(X, Y, Y_api, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)

    plt.show()

# %%
def paint_Neff_F(Neff, taua, F, mu0=2.81, sigma0=0.12):
    Neff = np.array(Neff)
    taua = np.array(taua)
    F = np.array(F)

    plt.figure(figsize=(8,6))
    plt.imshow(Neff, 
               extent=[np.log10(taua[0]), np.log10(taua[-1]), np.log10(F[0]), np.log10(F[-1])], 
               origin='lower', aspect='auto', cmap='viridis',
               interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$N_{\rm eff}$', fontsize=16, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    plt.xlabel(r'$\tau_a (s)$', fontsize=14)
    plt.ylabel(r'$f_{\tau_a}$', fontsize=14)
    plt.title(r'$N_{\rm eff}$ for different $\tau_a$ and $f_{\tau_a}$', fontsize=16)

    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]
    tick_positions = np.log10(tick_vals)
    plt.xticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)

    # --- MARCAR VALORES RECHAZADOS ---
    z = (Neff - mu0)/sigma0
    p_two = 2 * stats.norm.sf(np.abs(z))

    # Coordenadas de los puntos rechazados
    X, Y = np.meshgrid(np.log10(taua), np.log10(F))

    # Contornos opcionales (por ejemplo, niveles de interés)
    CS = plt.contour(X, Y, p_two, levels = [0.05],colors='red', linestyles='--', linewidths=2)
    plt.contourf(X, Y, Neff, levels=[mu0-sigma0, mu0+sigma0], colors='none',
             hatches=['---'], alpha=0)
    plt.show()

#%%
def paint_Neff_m_taua (Neff, taua, m, mu0, sigma0):
    Neff = np.array(Neff)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Neff, 
            extent=[np.log10(m[0]), np.log10(m[-1]), np.log10(taua[0]), np.log10(taua[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')

    cbar = plt.colorbar()
    cbar.set_label(label=r'$N_{\mathrm{eff}}$', 
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar

    plt.xlabel(r'$m_a$ (MeV)', fontsize = 14)
    plt.ylabel(r'$\tau_a (s)$', fontsize = 14)
    plt.title(rf'$N_{{\mathrm{{eff}}}}$ for $\tau_a$ and $\rho_a/\rho_{{SM}} (T_0)$',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.yticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])
    tick_vals_x = [300,400,500,600,700,800,900]
    tick_positions_x = np.log10(tick_vals_x)  # posiciones en escala log10

    plt.xticks(tick_positions_x, [str(v) for v in tick_vals_x])
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    z = (Neff - mu0)/sigma0
    p_two = 2 * stats.norm.sf(np.abs(z))

    # Coordenadas de los puntos rechazados
    X, Y = np.meshgrid(np.log10(m), np.log10(taua))

    # Contornos opcionales (por ejemplo, niveles de interés)
    CS = plt.contour(X, Y, p_two, levels = [0.05],colors='red', linestyles='--', linewidths=2)
    plt.contourf(X, Y, Neff, levels=[mu0-sigma0, mu0+sigma0], colors='none',
             hatches=['---'], alpha=0)
    plt.show()

def paint_Xn_pions_m_taua (Y_api, taua, m):
    Y_api = np.array(Y_api)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Y_api, 
            extent=[np.log10(m[0]), np.log10(m[-1]),np.log10(taua[0]), np.log10(taua[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
        interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\frac{X_n^{(a + \pi)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$', 
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar

    plt.xlabel(r'$m_a$ (MeV)', fontsize = 14)
    plt.ylabel(r'$\tau_a (s)$', fontsize = 14)
    plt.title(rf'$\frac{{X_n^{{(a + \pi)}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ ',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.yticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])

    tick_vals_x = [300,400,500,600,700,800,900]
    tick_positions_x = np.log10(tick_vals_x)  # posiciones en escala log10

    plt.xticks(tick_positions_x, [str(v) for v in tick_vals_x])    
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    Y= np.log10(taua)
    X = np.log10(m)
    CS = plt.contour(X, Y, Y_api, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)
    plt.show()
    plt.show()
# %%
def paint_Xn_no_pions_m_taua (Y_a, taua, m):
    Y_a = np.array(Y_a)  # asegurar que es numpy array de shape (len(prop), len(taua))

    plt.figure(figsize=(8,6))
    plt.imshow(Y_a, 
            extent=[np.log10(m[0]), np.log10(m[-1]),np.log10(taua[0]), np.log10(taua[-1])], 
            origin='lower', aspect='auto', cmap='viridis',
           interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label(label=r'$\frac{X_n^{(a)} -X_n^0 }{X_n^0}|_{T = 78 KeV}$',
        fontsize=16,      # tamaño del label
        labelpad=15       # separa el label del colorbar (mueve a la derecha)
    )
    cbar.ax.tick_params(labelsize=12)  # aumenta el tamaño de los números de la colorbar

    plt.xlabel(r'$m_a$ (MeV)', fontsize = 14)
    plt.ylabel(r'$\tau_a (s)$', fontsize = 14)
    plt.title(rf'$\frac{{X_n^{{(a)}} -X_n^0 }}{{X_n^0}}|_{{T = 78 KeV}}$ ',
        fontsize=16,
        y=1.05,
        loc='center'
    )
    def log_tick_formatter_y(val, pos=None):
        exponent = int(val)
        return rf"$10^{{{exponent}}}$"

    tick_vals = [0.01, 0.1, 1, 10]  # en unidades reales de taua

    # Posiciones en el eje log10 (coincide con extent de imshow)
    tick_positions = np.log10(tick_vals)

    plt.yticks(tick_positions, [f"$10^{{{int(np.log10(v))}}}$" for v in tick_vals])

    tick_vals_x = [300,400,500,600,700,800,900]
    tick_positions_x = np.log10(tick_vals_x)  # posiciones en escala log10

    plt.xticks(tick_positions_x, [str(v) for v in tick_vals_x])
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 12)

    Y = np.log10(taua)
    X = np.log10(m)
    CS = plt.contour(X, Y, Y_a, levels=[-0.02,0.02], colors='red', linestyles='--', linewidths=2)
    plt.show()

#%%
def excluded_ma_taua (Y_api, Y_a, dat_n, ma, taua, mu0, sigma0 ):
    X,Y = np.meshgrid(ma, taua)

    z = (dat_n - mu0)/sigma0
    p_two = 2 * stats.norm.sf(np.abs(z))
    # Región roja
    #z_red = np.where(mask_red, 1, 0)
    plt.contour(X, Y, p_two, levels=[0.05], colors='red', linewidths=2, zorder=3)
    plt.contourf(X, Y, p_two, levels=[0,0.05], colors='red', alpha=0.2)
    plt.contourf(X, Y, dat_n, levels=[mu0-sigma0, mu0+sigma0], colors='none',
                hatches=['---'], alpha=0)

    # Región verde
    #mask_Ya = dat_ya>abs(0.01)
    #z_green = np.where(mask_Ya, 1, 0)
    plt.contour(X, Y, Y_api, levels=[-0.02,0.02], colors='blue', linewidths=2, zorder=5)
    plt.contourf(X, Y, Y_api, levels=[-np.inf,-0.02], colors='blue', alpha=0.2)
    plt.contourf(X, Y, Y_api, levels=[0.02,np.inf], colors='blue', alpha=0.2)


    # Leyenda
    legend_elements = [
        Patch(facecolor='#e41a1c', alpha=1, label=r'$N_{\rm eff}$'),
        Patch(facecolor='blue', alpha=1, label=r'$Y_p$'),
    ]

    # Escalas y etiquetas
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_a(s)$', fontsize=16)
    plt.ylabel(r'$\tau_a(s)$', fontsize=16)

    # Leyenda a la derecha
    plt.legend(handles=legend_elements, loc='center left',
            bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14)

    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize = 14)
    plt.tight_layout()
    plt.show()