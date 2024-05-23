#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:59:18 2024

@author: johnpaulmbagwu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# MATERIAL CONSTANTS
proton_mass = 938.28


# Fluorine (F)
Z_F = 9 
A_F = 18.9984 # g/mol
I_F = 115.0 # eV
A1_F = 2.085
A2_F = 2.352
A3_F = 2157
A4_F = 2634
A5_F = 0.01816
C1 = 0.32
C2 = 0.33
X1 = 4.4096
X0 = 1.8433
a = 0.11083
m = 3.2962
C = 10.9653

# Barium (Ba)
Z_Ba = 56
A_Ba = 137.33 # g/mol  
I_Ba = 491.0 # eV
A1_Ba = 7.899
A2_Ba = 8.911
A3_Ba = 12430
A4_Ba = 402.1
A5_Ba = 0.004511
C1 = 0.41
C2 = 0.45
X1 = 3.4547
X0 = 0.4190
a = 0.18268
m = 2.8906
C = 6.3153

# Bromine (Br)
Z_Br = 35
A_Br = 79.904 # g/mol
I_Br = 343.0 # eV
A1_Br = 6.658
A2_Br = 7.536
A3_Br = 7694
A4_Br = 222.3
A5_Br = 0.006509
C1 = 0.45
C2 = 0.50
X1 = 4.9899
X0 = 1.5262
a = 0.06335
m = 3.4670
C = 11.7307

# Define common functions
def proton_beta(kinetic_energy):
    return (1 - (1 / ((kinetic_energy / proton_mass) + 1)) ** 2) ** 0.5

def bloch_correction(kinetic_energy):
    be = proton_beta(kinetic_energy)
    y = 1 / (137 * be)
    bloch = -y**2 * (1.202 - y**2 * (1.042 - 0.855 * y**2 + 0.343 * y**4))
    return bloch

def shell_correction(kinetic_energy, Z, C1, C2):
    v = proton_beta(kinetic_energy)
    C = C1 * v * np.exp(-C2 * v)
    return C

def barkas_correction(T, Z):
    T_eV = T * 1000
    L_low = 0.001 * T_eV
    L_high = (1.5 / (T_eV**0.4)) + (45000 / Z) / (T_eV**1.6)
    barkas = L_low * L_high / (L_low + L_high)
    return barkas

def proton_mass_collisional_sp(kinetic_energy, Z, A, I):
    factor = 0.3071 * Z / (A * (proton_beta(kinetic_energy)) ** 2)
    factor2 = (13.8373 + np.log((proton_beta(kinetic_energy)) ** 2 / (1 - (proton_beta(kinetic_energy)) ** 2)) - (proton_beta(kinetic_energy)) ** 2 - np.log(I))
    return factor * factor2

def density_corrections(kinetic_energy, C, X0, X1, a, m):
    gamma = (1 - (proton_beta(kinetic_energy)) ** 2) ** (-0.5)
    X = np.log10(proton_beta(kinetic_energy) * gamma)
    if X < X0:
        return 0
    elif X > X0 and X < X1:
        return 4.6052 * X + a * (X1 - X) ** m + C
    elif X > X1:
        return 4.6052 * X + C

def dedx(kinetic_energy, I, Z, X0, X1, a, m, C, A1, A2, A3, A4, A5):
    bloch = bloch_correction(kinetic_energy)
    shell = shell_correction(kinetic_energy, Z, C1, C2)
    barkas = barkas_correction(kinetic_energy, Z)
    SP = proton_mass_collisional_sp(kinetic_energy, Z, A_Br, I)
    SP -= shell / Z
    SP += barkas + bloch
    SP -= 0.00002 * density_corrections(kinetic_energy, C, X0, X1, a, m)
    return SP

def low_cross(kinetic_energy, A1, A2, A3, A4, A5):
    T_s = 1000 * kinetic_energy / 1.0073
    if T_s <= 10:
        return A1 * T_s ** 0.5
    else:
        elow = A2 * T_s ** 0.45
        ehigh = (A3 / T_s) * np.log(1 + (A4 / T_s) + A5 * T_s)
        low_cross_fac = (ehigh * elow) / (ehigh + elow)
        return low_cross_fac

# Energy axis stopping at 1 MeV
x_axis = np.linspace(1, 10000, 100000)

# Stopping power axis
y_axis_BaFBr = np.zeros(np.size(x_axis))

# Calculate stopping power for BaFBr compound
for nenergies in range(np.size(x_axis)):
    y_axis_BaFBr[nenergies] = dedx(x_axis[nenergies], I_F, Z_F, X0, X1, a, m, C, A1_F, A2_F, A3_F, A4_F, A5_F) + \
                               dedx(x_axis[nenergies], I_Ba, Z_Ba, X0, X1, a, m, C, A1_Ba, A2_Ba, A3_Ba, A4_Ba, A5_Ba) + \
                               dedx(x_axis[nenergies], I_Br, Z_Br, X0, X1, a, m, C, A1_Br, A2_Br, A3_Br, A4_Br, A5_Br)

# Add the Low Energy Stopping Power plot
x_axis_low = np.linspace(0.001, 1, 2000)
y_axis_BaFBr_low = np.zeros(np.size(x_axis_low))

for nenergies in range(np.size(x_axis_low)):
    y_axis_BaFBr_low[nenergies] = low_cross(x_axis_low[nenergies], A1_F, A2_F, A3_F, A4_F, A5_F) + \
                                   low_cross(x_axis_low[nenergies], A1_Ba, A2_Ba, A3_Ba, A4_Ba, A5_Ba) + \
                                   low_cross(x_axis_low[nenergies], A1_Br, A2_Br, A3_Br, A4_Br, A5_Br)

multiplicative_factor_BaFBr = y_axis_BaFBr[0] / y_axis_BaFBr_low[-1]
for nenergies in range(np.size(x_axis_low)):
    y_axis_BaFBr_low[nenergies] = y_axis_BaFBr_low[nenergies] * multiplicative_factor_BaFBr

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 8))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Plot Mass Collisional Stopping Power
plt.semilogy(x_axis, y_axis_BaFBr, label='BaFBr Stopping Power', color='b')

# Plot Low Energy Stopping Power
plt.semilogy(x_axis_low, y_axis_BaFBr_low, label='BaFBr Low Energy Stopping Power', color='black', linewidth=2.5)

plt.xlabel('Proton Energy (MeV)')
plt.ylabel('Stopping Power (MeV cm$^2$/g)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-3, 1e4)
plt.ylim(1, 1e4)
plt.grid(True)
plt.show()