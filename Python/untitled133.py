#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:16:31 2024

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

# Anderson-Ziegler Fluorine Parameters
A1_F = 2.085
A2_F = 2.352
A3_F = 2157
A4_F = 2634
A5_F = 0.01816

# Fluorine Density Correction Parameters  
X1_F = 4.4096
X0_F = 1.8433
a_F = 0.11083
m_F = 3.2962
C_F = 10.9653

# Fluorine Shell Correction Parameters
C1_F = 0.32
C2_F = 0.33

# Barium (Ba)
Z_Ba = 56
A_Ba = 137.33 # g/mol
I_Ba = 491.0 # eV

# Anderson-Ziegler Barium Parameters
A1_Ba = 7.899
A2_Ba = 8.911
A3_Ba = 12430
A4_Ba = 402.1
A5_Ba = 0.004511

# Barium Density Correction Parameters
X1_Ba = 3.4547
X0_Ba = 0.4190
a_Ba = 0.18268
m_Ba = 2.8906
C_Ba = 6.3153

# Barium Shell Correction Parameters
C1_Ba = 0.41
C2_Ba = 0.45

# Bromine (Br)
Z_Br = 35
A_Br = 79.904 # g/mol
I_Br = 343.0 # eV

# Anderson-Ziegler Bromine Parameters
A1_Br = 6.658
A2_Br = 7.536
A3_Br = 7694
A4_Br = 222.3
A5_Br = 0.006509

# Bromine Density Correction Parameters
X1_Br = 4.9899
X0_Br = 1.5262
a_Br = 0.06335
m_Br = 3.4670
C_Br = 11.7307

# Bromine Shell Correction Parameters
C1_Br = 0.45
C2_Br = 0.50

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

def dedx(kinetic_energy, I, Z, A, X0, X1, a, m, C, C1, C2):
    bloch = bloch_correction(kinetic_energy)
    shell = shell_correction(kinetic_energy, Z, C1, C2)
    barkas = barkas_correction(kinetic_energy, Z)
    SP = proton_mass_collisional_sp(kinetic_energy, Z, A, I)
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

def compute_stopping_power(Z, A, I, X0, X1, a, m, C, C1, C2, A1, A2, A3, A4, A5):
    x_axis = np.linspace(1, 10000, 100000)
    y_axis = np.zeros(np.size(x_axis))
    for nenergies in range(np.size(x_axis)):
        y_axis[nenergies] = dedx(x_axis[nenergies], I, Z, A, X0, X1, a, m, C, C1, C2)

    x_axis_low = np.linspace(0.001, 1, 2000)
    y_axis_low = np.zeros(np.size(x_axis_low))
    for nenergies in range(np.size(x_axis_low)):
        y_axis_low[nenergies] = low_cross(x_axis_low[nenergies], A1, A2, A3, A4, A5)

    multiplicative_factor = y_axis[0] / y_axis_low[-1]
    for nenergies in range(np.size(x_axis_low)):
        y_axis_low[nenergies] *= multiplicative_factor

    return x_axis, y_axis, x_axis_low, y_axis_low

# Compute stopping power for Fluorine, Barium, and Bromine
x_axis_F, y_axis_F, x_axis_F_low, y_axis_F_low = compute_stopping_power(Z_F, A_F, I_F, X0_F, X1_F, a_F, m_F, C_F, C1_F, C2_F, A1_F, A2_F, A3_F, A4_F, A5_F)
x_axis_Ba, y_axis_Ba, x_axis_Ba_low, y_axis_Ba_low = compute_stopping_power(Z_Ba, A_Ba, I_Ba, X0_Ba, X1_Ba, a_Ba, m_Ba, C_Ba, C1_Ba, C2_Ba, A1_Ba, A2_Ba, A3_Ba, A4_Ba, A5_Ba)
x_axis_Br, y_axis_Br, x_axis_Br_low, y_axis_Br_low = compute_stopping_power(Z_Br, A_Br, I_Br, X0_Br, X1_Br, a_Br, m_Br, C_Br, C1_Br, C2_Br, A1_Br, A2_Br, A3_Br, A4_Br, A5_Br)

# Weighted averages
fraction_F = A_F / (A_Ba + A_F + A_Br)
fraction_Ba = A_Ba / (A_Ba + A_F + A_Br)
fraction_Br = A_Br / (A_Ba + A_F + A_Br)

# Interpolating to match sizes of arrays for each element
y_axis_Ba_interp = np.interp(x_axis_F, x_axis_Ba, y_axis_Ba)
y_axis_Br_interp = np.interp(x_axis_F, x_axis_Br, y_axis_Br)

y_axis_Ba_low_interp = np.interp(x_axis_F_low, x_axis_Ba_low, y_axis_Ba_low)
y_axis_Br_low_interp = np.interp(x_axis_F_low, x_axis_Br_low, y_axis_Br_low)

# Combining the stopping powers
y_axis_BaFBr = fraction_F * y_axis_F + fraction_Ba * y_axis_Ba_interp + fraction_Br * y_axis_Br_interp
y_axis_BaFBr_low = fraction_F * y_axis_F_low + fraction_Ba * y_axis_Ba_low_interp + fraction_Br * y_axis_Br_low_interp

# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(x_axis_F, y_axis_BaFBr, label='BaFBr Stopping Power', color='b')
plt.plot(x_axis_F_low, y_axis_BaFBr_low, label='BaFBr Low Energy Stopping Power', color='r', linewidth=2.5)

plt.xlabel('Proton Energy (MeV)')
plt.ylabel('Stopping Power (MeV cm$^2$/g)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-3, 1e4)
plt.ylim(1, 1e4)
plt.grid(True)
plt.show()