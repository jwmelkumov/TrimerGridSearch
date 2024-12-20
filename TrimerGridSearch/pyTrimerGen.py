#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 John W. Melkumov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Author: John W. Melkumov
# Date: 06.01.2024.
# Description:
# TrimerGen takes as input three XYZ files (See Ref. * below.)            
# with each file containing atom labels and coordinates of one of          
# three monomers, A, B, and C, as well as the desired center-of-mass to          
# center-of-mass separation between each pair of monomers in angstroms, 
# and 8 Euler angles in the ZYZ convention.                                            
# Note: Although 9 Euler angles are used, (alphaA, betaA, gammaA) 
# (alphaB, betaB, gammaB), and (alphaC, betaC, gammaC), alphaA is 
# hardcoded and set to 0.
# Ref.:                                                                    
# * www.ccl.net/chemistry/resources/messages/1996/10/21.005-dir/index.html 
# ----------------------------------------------------------------------------------------

import os
import sys
import numpy as np

def getCOM(coords, masses):
    total_mass = sum(masses)
    com_x = sum(coords[:, 0] * masses) / total_mass
    com_y = sum(coords[:, 1] * masses) / total_mass
    com_z = sum(coords[:, 2] * masses) / total_mass
    return np.array([com_x, com_y, com_z])

def compute_inertia_tensor(xyz, masses):
    inertia_tensor = np.zeros((3, 3))
    for i in range(len(masses)):
        r = xyz[i]
        mass = masses[i]
        inertia_tensor += mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return inertia_tensor

def euler_rotation_matrix(alpha, beta, gamma):
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    Rz_alpha = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad), 0],
                         [np.sin(alpha_rad), np.cos(alpha_rad), 0],
                         [0, 0, 1]])

    Ry_beta = np.array([[np.cos(beta_rad), 0, np.sin(beta_rad)],
                        [0, 1, 0],
                        [-np.sin(beta_rad), 0, np.cos(beta_rad)]])

    Rz_gamma = np.array([[np.cos(gamma_rad), -np.sin(gamma_rad), 0],
                         [np.sin(gamma_rad), np.cos(gamma_rad), 0],
                         [0, 0, 1]])

    return np.dot(Rz_gamma, np.dot(Ry_beta, Rz_alpha))

def TrimerGen(xyzfileA, xyzfileB, xyzfileC, sepAB, sepBC, sepAC, betaA, gammaA, alphaB, betaB, gammaB, alphaC, betaC, gammaC):
    alphaA = 0.0
    sepAB = float(sepAB)
    sepBC = float(sepBC)
    sepAC = float(sepAC)
    betaA = float(betaA)
    gammaA = float(gammaA)
    alphaB = float(alphaB)
    betaB = float(betaB)
    gammaB = float(gammaB)
    alphaC = float(alphaC)
    betaC = float(betaC)
    gammaC = float(gammaC)
    monAfile = xyzfileA
    monBfile = xyzfileB
    monCfile = xyzfileC
    labelsA = []
    xyzA = []
    labelsB = []
    xyzB = []
    labelsC = []
    xyzC = []
    with open(monAfile, 'r') as f:
        next(f)
        next(f)
        for line in f:
            if len(line) >= 4:
                col = line.split()
                labelsA.append(col[0]) 
                floats = [float(col[i]) for i in range(1,4)]
                xyzA.append(floats)
    with open(monBfile, 'r') as f:
        next(f)
        next(f)
        for line in f:
            if len(line) >= 4:
                col = line.split()
                labelsB.append(col[0]) 
                floats = [float(col[i]) for i in range(1,4)]
                xyzB.append(floats)
    with open(monCfile, 'r') as f:
        next(f)
        next(f)
        for line in f:
            if len(line) >= 4:
                col = line.split()
                labelsC.append(col[0]) 
                floats = [float(col[i]) for i in range(1,4)]
                xyzC.append(floats)

    massdict = {
    'O': 16.000, 
    'H': 1.000, 
    'C': 12.000, 
    'N': 14.000, 
    'M': 0.000, # TIP4P Off-Atomic Site
    'L': 0.000, # Force Field "Lone Pair"
    'He': 4.003,
    'Li': 6.940,
    'Be': 9.012,
    'B': 10.810,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.085,
    'P': 30.974,
    'S': 32.060,
    'Cl': 35.450,
    'Ar': 39.948,
    'K': 39.098,
    'Ca': 40.078,
    'Sc': 44.956,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Co': 58.933,
    'Ni': 58.693,
    'Cu': 63.546,
    'Zn': 65.380,
    'Ga': 69.723,
    'Ge': 72.630,
    'As': 74.922,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.468,
    'Sr': 87.620,
    'Y': 88.906,
    'Zr': 91.224,
    'Nb': 92.906,
    'Mo': 95.950,
    'Tc': 98.000,
    'Ru': 101.070,
    'Rh': 102.906,
    'Pd': 106.420,
    'Ag': 107.868,
    'Cd': 112.414,
    'In': 114.818,
    'Sn': 118.710,
    'Sb': 121.760,
    'Te': 127.600,
    'I': 126.904,
    'Xe': 131.293,
    'Cs': 132.905,
    'Ba': 137.327,
    'La': 138.905,
    'Ce': 140.116,
    'Pr': 140.907,
    'Nd': 144.242,
    'Pm': 145.000,
    'Sm': 150.360,
    'Eu': 151.964,
    'Gd': 157.250,
    'Tb': 158.925,
    'Dy': 162.500,
    'Ho': 164.930,
    'Er': 167.259,
    'Tm': 168.934,
    'Yb': 173.045,
    'Lu': 174.966,
    'Hf': 178.490,
    'Ta': 180.948,
    'W': 183.840,
    'Re': 186.207,
    'Os': 190.230,
    'Ir': 192.217,
    'Pt': 195.084,
    'Au': 196.967,
    'Hg': 200.592,
    'Tl': 204.383,
    'Pb': 207.200,
    'Bi': 208.980,
    'Po': 209.000,
    'At': 210.000,
    'Rn': 222.000,
    'Fr': 223.000,
    'Ra': 226.000,
    'Ac': 227.000,
    'Th': 232.038,
    'Pa': 231.036,
    'U': 238.029,
    'Np': 237.000,
    'Pu': 244.000,
    'Am': 243.000,
    'Cm': 247.000,
    'Bk': 247.000,
    'Cf': 251.000,
    'Es': 252.000,
    'Fm': 257.000,
    'Md': 258.000,
    'No': 259.000,
    'Lr': 266.000,
    'Rf': 267.000,
    'Db': 270.000,
    'Sg': 271.000,
    'Bh': 270.000,
    'Hs': 269.000,
    'Mt': 278.000,
    'Ds': 281.000,
    'Rg': 282.000,
    'Cn': 285.000,
    'Nh': 286.000,
    'Fl': 289.000,
    'Mc': 290.000,
    'Lv': 293.000,
    'Ts': 294.000,
    'Og': 294.000,
    }

    massA = []
    for i in labelsA:
        if (i[0] == 'O'):
            massA.append(massdict['O'])
        elif (i[0] == 'H'):
            massA.append(massdict['H'])
        elif (i[0] == 'C'):
            massA.append(massdict['C'])
        elif (i[0] == 'N'):
            massA.append(massdict['N'])
        elif (i[0] == 'M'):
            massA.append(massdict['M'])
        elif (i[0] == 'L') and i[1].isupper():
            massA.append(massdict['L'])
        elif (i[0] == 'B'):
            massA.append(massdict['B'])
        elif (i[0] == 'F'):
            massA.append(massdict['F'])
        elif (i[0] == 'P'):
            massA.append(massdict['P'])
        elif (i[0] == 'S'):
            massA.append(massdict['S'])
        elif (i[0] == 'K'):
            massA.append(massdict['K'])
        elif (i[0] == 'V'):
            massA.append(massdict['V'])
        elif (i[0] == 'Y'):
            massA.append(massdict['Y'])
        elif (i[0] == 'I'):
            massA.append(massdict['I'])
        elif (i[0] == 'W'):
            massA.append(massdict['W'])
        elif (i[0] == 'U'):
            massA.append(massdict['U'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'He':
                massA.append(massdict['He'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Li':
                massA.append(massdict['Li'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Be':
                massA.append(massdict['Be'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ne':
                massA.append(massdict['Ne'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Na':
                massA.append(massdict['Na'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Mg':
                massA.append(massdict['Mg'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Al':
                massA.append(massdict['Al'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Si':
                massA.append(massdict['Si'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cl':
                massA.append(massdict['Cl'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ar':
                massA.append(massdict['Ar'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ca':
                massA.append(massdict['Ca'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sc':
                massA.append(massdict['Sc'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ti':
                massA.append(massdict['Ti'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cr':
                massA.append(massdict['Cr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Mn':
                massA.append(massdict['Mn'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Fe':
                massA.append(massdict['Fe'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Co':
                massA.append(massdict['Co'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ni':
                massA.append(massdict['Ni'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cu':
                massA.append(massdict['Cu'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Zn':
                massA.append(massdict['Zn'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ga':
                massA.append(massdict['Ga'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ge':
                massA.append(massdict['Ge'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'As':
                massA.append(massdict['As'])            
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Se':
                massA.append(massdict['Se'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Br':
                massA.append(massdict['Br'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Kr':
                massA.append(massdict['Kr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Rb':
                massA.append(massdict['Rb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sr':
                massA.append(massdict['Sr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Zr':
                massA.append(massdict['Zr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Nb':
                massA.append(massdict['Nb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Mo':
                massA.append(massdict['Mo'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Tc':
                massA.append(massdict['Tc'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ru':
                massA.append(massdict['Ru'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Rh':
                massA.append(massdict['Rh'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pd':
                massA.append(massdict['Pd'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ag':
                massA.append(massdict['Ag'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cd':
                massA.append(massdict['Cd'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'In':
                massA.append(massdict['In'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sn':
                massA.append(massdict['Sn'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sb':
                massA.append(massdict['Sb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Te':
                massA.append(massdict['Te'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Xe':
                massA.append(massdict['Xe'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cs':
                massA.append(massdict['Cs'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ba':
                massA.append(massdict['Ba'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'La':
                massA.append(massdict['La'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ce':
                massA.append(massdict['Ce'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pr':
                massA.append(massdict['Pr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Nd':
                massA.append(massdict['Nd'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pm':
                massA.append(massdict['Pm'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sm':
                massA.append(massdict['Sm'])           
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Eu':
                massA.append(massdict['Eu'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Gd':
                massA.append(massdict['Gd'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Tb':
                massA.append(massdict['Tb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Dy':
                massA.append(massdict['Dy'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ho':
                massA.append(massdict['Ho'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Er':
                massA.append(massdict['Er'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Tm':
                massA.append(massdict['Tm'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Yb':
                massA.append(massdict['Yb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Lu':
                massA.append(massdict['Lu'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Hf':
                massA.append(massdict['Hf'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ta':
                massA.append(massdict['Ta'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Re':
                massA.append(massdict['Re'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Os':
                massA.append(massdict['Os'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ir':
                massA.append(massdict['Ir'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pt':
                massA.append(massdict['Pt'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Au':
                massA.append(massdict['Au'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Hg':
                massA.append(massdict['Hg'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Tl':
                massA.append(massdict['Tl'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pb':
                massA.append(massdict['Pb'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Bi':
                massA.append(massdict['Bi'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Po':
                massA.append(massdict['Po'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'At':
                massA.append(massdict['At'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Rn':
                massA.append(massdict['Rn'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Fr':
                massA.append(massdict['Fr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ra':
                massA.append(massdict['Ra'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ac':
                massA.append(massdict['Ac'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Th':
                massA.append(massdict['Th'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pa':
                massA.append(massdict['Pa'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Np':
                massA.append(massdict['Np'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Pu':
                massA.append(massdict['Pu'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Am':
                massA.append(massdict['Am'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cm':
                massA.append(massdict['Cm'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Bk':
                massA.append(massdict['Bk'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cf':
                massA.append(massdict['Cf'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Es':
                massA.append(massdict['Es'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Fm':
                massA.append(massdict['Fm'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Md':
                massA.append(massdict['Md'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'No':
                massA.append(massdict['No'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Lr':
                massA.append(massdict['Lr'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Rf':
                massA.append(massdict['Rf'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Db':
                massA.append(massdict['Db'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Sg':
                massA.append(massdict['Sg'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Bh':
                massA.append(massdict['Bh'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Hs':
                massA.append(massdict['Hs'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Mt':
                massA.append(massdict['Mt'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ds':
                massA.append(massdict['Ds'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Rg':
                massA.append(massdict['Rg'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Cn':
                massA.append(massdict['Cn'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Nh':
                massA.append(massdict['Nh'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Fl':
                massA.append(massdict['Fl'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Mc':
                massA.append(massdict['Mc'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Lv':
                massA.append(massdict['Lv'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Ts':
                massA.append(massdict['Ts'])
        elif len(labelsA) > 1 and i[1].islower():
            if i[:2] == 'Og':
                massA.append(massdict['Og'])
    massB = []
    for i in labelsB:
        if (i[0] == 'O'):
            massB.append(massdict['O'])
        elif (i[0] == 'H'):
            massB.append(massdict['H'])
        elif (i[0] == 'C'):
            massB.append(massdict['C'])
        elif (i[0] == 'N'):
            massB.append(massdict['N'])
        elif (i[0] == 'M'):
            massB.append(massdict['M'])
        elif (i[0] == 'L') and i[1].isupper():
            massB.append(massdict['L'])
        elif (i[0] == 'B'):
            massB.append(massdict['B'])
        elif (i[0] == 'F'):
            massB.append(massdict['F'])
        elif (i[0] == 'P'):
            massB.append(massdict['P'])
        elif (i[0] == 'S'):
            massB.append(massdict['S'])
        elif (i[0] == 'K'):
            massB.append(massdict['K'])
        elif (i[0] == 'V'):
            massB.append(massdict['V'])
        elif (i[0] == 'Y'):
            massB.append(massdict['Y'])
        elif (i[0] == 'I'):
            massB.append(massdict['I'])
        elif (i[0] == 'W'):
            massB.append(massdict['W'])
        elif (i[0] == 'U'):
            massB.append(massdict['U'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'He':
                massB.append(massdict['He'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Li':
                massB.append(massdict['Li'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Be':
                massB.append(massdict['Be'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ne':
                massB.append(massdict['Ne'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Na':
                massB.append(massdict['Na'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Mg':
                massB.append(massdict['Mg'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Al':
                massB.append(massdict['Al'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Si':
                massB.append(massdict['Si'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cl':
                massB.append(massdict['Cl'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ar':
                massB.append(massdict['Ar'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ca':
                massB.append(massdict['Ca'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sc':
                massB.append(massdict['Sc'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ti':
                massB.append(massdict['Ti'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cr':
                massB.append(massdict['Cr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Mn':
                massB.append(massdict['Mn'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Fe':
                massB.append(massdict['Fe'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Co':
                massB.append(massdict['Co'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ni':
                massB.append(massdict['Ni'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cu':
                massB.append(massdict['Cu'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Zn':
                massB.append(massdict['Zn'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ga':
                massB.append(massdict['Ga'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ge':
                massB.append(massdict['Ge'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'As':
                massB.append(massdict['As'])            
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Se':
                massB.append(massdict['Se'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Br':
                massB.append(massdict['Br'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Kr':
                massB.append(massdict['Kr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Rb':
                massB.append(massdict['Rb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sr':
                massB.append(massdict['Sr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Zr':
                massB.append(massdict['Zr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Nb':
                massB.append(massdict['Nb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Mo':
                massB.append(massdict['Mo'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Tc':
                massB.append(massdict['Tc'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ru':
                massB.append(massdict['Ru'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Rh':
                massB.append(massdict['Rh'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pd':
                massB.append(massdict['Pd'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ag':
                massB.append(massdict['Ag'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cd':
                massB.append(massdict['Cd'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'In':
                massB.append(massdict['In'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sn':
                massB.append(massdict['Sn'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sb':
                massB.append(massdict['Sb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Te':
                massB.append(massdict['Te'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Xe':
                massB.append(massdict['Xe'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cs':
                massB.append(massdict['Cs'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ba':
                massB.append(massdict['Ba'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'La':
                massB.append(massdict['La'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ce':
                massB.append(massdict['Ce'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pr':
                massB.append(massdict['Pr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Nd':
                massB.append(massdict['Nd'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pm':
                massB.append(massdict['Pm'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sm':
                massB.append(massdict['Sm'])           
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Eu':
                massB.append(massdict['Eu'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Gd':
                massB.append(massdict['Gd'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Tb':
                massB.append(massdict['Tb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Dy':
                massB.append(massdict['Dy'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ho':
                massB.append(massdict['Ho'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Er':
                massB.append(massdict['Er'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Tm':
                massB.append(massdict['Tm'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Yb':
                massB.append(massdict['Yb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Lu':
                massB.append(massdict['Lu'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Hf':
                massB.append(massdict['Hf'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ta':
                massB.append(massdict['Ta'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Re':
                massB.append(massdict['Re'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Os':
                massB.append(massdict['Os'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ir':
                massB.append(massdict['Ir'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pt':
                massB.append(massdict['Pt'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Au':
                massB.append(massdict['Au'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Hg':
                massB.append(massdict['Hg'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Tl':
                massB.append(massdict['Tl'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pb':
                massB.append(massdict['Pb'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Bi':
                massB.append(massdict['Bi'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Po':
                massB.append(massdict['Po'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'At':
                massB.append(massdict['At'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Rn':
                massB.append(massdict['Rn'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Fr':
                massB.append(massdict['Fr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ra':
                massB.append(massdict['Ra'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ac':
                massB.append(massdict['Ac'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Th':
                massB.append(massdict['Th'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pa':
                massB.append(massdict['Pa'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Np':
                massB.append(massdict['Np'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Pu':
                massB.append(massdict['Pu'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Am':
                massB.append(massdict['Am'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cm':
                massB.append(massdict['Cm'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Bk':
                massB.append(massdict['Bk'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cf':
                massB.append(massdict['Cf'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Es':
                massB.append(massdict['Es'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Fm':
                massB.append(massdict['Fm'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Md':
                massB.append(massdict['Md'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'No':
                massB.append(massdict['No'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Lr':
                massB.append(massdict['Lr'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Rf':
                massB.append(massdict['Rf'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Db':
                massB.append(massdict['Db'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Sg':
                massB.append(massdict['Sg'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Bh':
                massB.append(massdict['Bh'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Hs':
                massB.append(massdict['Hs'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Mt':
                massB.append(massdict['Mt'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ds':
                massB.append(massdict['Ds'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Rg':
                massB.append(massdict['Rg'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Cn':
                massB.append(massdict['Cn'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Nh':
                massB.append(massdict['Nh'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Fl':
                massB.append(massdict['Fl'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Mc':
                massB.append(massdict['Mc'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Lv':
                massB.append(massdict['Lv'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Ts':
                massB.append(massdict['Ts'])
        elif len(labelsB) > 1 and i[1].islower():
            if i[:2] == 'Og':
                massB.append(massdict['Og'])
    massC = []
    for i in labelsC:
        if (i[0] == 'O'):
            massC.append(massdict['O'])
        elif (i[0] == 'H'):
            massC.append(massdict['H'])
        elif (i[0] == 'C'):
            massC.append(massdict['C'])
        elif (i[0] == 'N'):
            massC.append(massdict['N'])
        elif (i[0] == 'M'):
            massC.append(massdict['M'])
        elif (i[0] == 'L') and i[1].isupper():
            massC.append(massdict['L'])
        elif (i[0] == 'B'):
            massC.append(massdict['B'])
        elif (i[0] == 'F'):
            massC.append(massdict['F'])
        elif (i[0] == 'P'):
            massC.append(massdict['P'])
        elif (i[0] == 'S'):
            massC.append(massdict['S'])
        elif (i[0] == 'K'):
            massC.append(massdict['K'])
        elif (i[0] == 'V'):
            massC.append(massdict['V'])
        elif (i[0] == 'Y'):
            massC.append(massdict['Y'])
        elif (i[0] == 'I'):
            massC.append(massdict['I'])
        elif (i[0] == 'W'):
            massC.append(massdict['W'])
        elif (i[0] == 'U'):
            massC.append(massdict['U'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'He':
                massC.append(massdict['He'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Li':
                massC.append(massdict['Li'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Be':
                massC.append(massdict['Be'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ne':
                massC.append(massdict['Ne'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Na':
                massC.append(massdict['Na'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Mg':
                massC.append(massdict['Mg'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Al':
                massC.append(massdict['Al'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Si':
                massC.append(massdict['Si'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cl':
                massC.append(massdict['Cl'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ar':
                massC.append(massdict['Ar'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ca':
                massC.append(massdict['Ca'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sc':
                massC.append(massdict['Sc'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ti':
                massC.append(massdict['Ti'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cr':
                massC.append(massdict['Cr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Mn':
                massC.append(massdict['Mn'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Fe':
                massC.append(massdict['Fe'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Co':
                massC.append(massdict['Co'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ni':
                massC.append(massdict['Ni'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cu':
                massC.append(massdict['Cu'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Zn':
                massC.append(massdict['Zn'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ga':
                massC.append(massdict['Ga'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ge':
                massC.append(massdict['Ge'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'As':
                massC.append(massdict['As'])            
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Se':
                massC.append(massdict['Se'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Br':
                massC.append(massdict['Br'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Kr':
                massC.append(massdict['Kr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Rb':
                massC.append(massdict['Rb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sr':
                massC.append(massdict['Sr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Zr':
                massC.append(massdict['Zr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Nb':
                massC.append(massdict['Nb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Mo':
                massC.append(massdict['Mo'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Tc':
                massC.append(massdict['Tc'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ru':
                massC.append(massdict['Ru'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Rh':
                massC.append(massdict['Rh'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pd':
                massC.append(massdict['Pd'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ag':
                massC.append(massdict['Ag'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cd':
                massC.append(massdict['Cd'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'In':
                massC.append(massdict['In'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sn':
                massC.append(massdict['Sn'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sb':
                massC.append(massdict['Sb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Te':
                massC.append(massdict['Te'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Xe':
                massC.append(massdict['Xe'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cs':
                massC.append(massdict['Cs'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ba':
                massC.append(massdict['Ba'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'La':
                massC.append(massdict['La'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ce':
                massC.append(massdict['Ce'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pr':
                massC.append(massdict['Pr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Nd':
                massC.append(massdict['Nd'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pm':
                massC.append(massdict['Pm'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sm':
                massC.append(massdict['Sm'])           
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Eu':
                massC.append(massdict['Eu'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Gd':
                massC.append(massdict['Gd'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Tb':
                massC.append(massdict['Tb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Dy':
                massC.append(massdict['Dy'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ho':
                massC.append(massdict['Ho'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Er':
                massC.append(massdict['Er'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Tm':
                massC.append(massdict['Tm'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Yb':
                massC.append(massdict['Yb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Lu':
                massC.append(massdict['Lu'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Hf':
                massC.append(massdict['Hf'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ta':
                massC.append(massdict['Ta'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Re':
                massC.append(massdict['Re'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Os':
                massC.append(massdict['Os'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ir':
                massC.append(massdict['Ir'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pt':
                massC.append(massdict['Pt'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Au':
                massC.append(massdict['Au'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Hg':
                massC.append(massdict['Hg'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Tl':
                massC.append(massdict['Tl'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pb':
                massC.append(massdict['Pb'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Bi':
                massC.append(massdict['Bi'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Po':
                massC.append(massdict['Po'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'At':
                massC.append(massdict['At'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Rn':
                massC.append(massdict['Rn'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Fr':
                massC.append(massdict['Fr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ra':
                massC.append(massdict['Ra'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ac':
                massC.append(massdict['Ac'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Th':
                massC.append(massdict['Th'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pa':
                massC.append(massdict['Pa'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Np':
                massC.append(massdict['Np'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Pu':
                massC.append(massdict['Pu'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Am':
                massC.append(massdict['Am'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cm':
                massC.append(massdict['Cm'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Bk':
                massC.append(massdict['Bk'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cf':
                massC.append(massdict['Cf'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Es':
                massC.append(massdict['Es'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Fm':
                massC.append(massdict['Fm'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Md':
                massC.append(massdict['Md'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'No':
                massC.append(massdict['No'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Lr':
                massC.append(massdict['Lr'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Rf':
                massC.append(massdict['Rf'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Db':
                massC.append(massdict['Db'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Sg':
                massC.append(massdict['Sg'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Bh':
                massC.append(massdict['Bh'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Hs':
                massC.append(massdict['Hs'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Mt':
                massC.append(massdict['Mt'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ds':
                massC.append(massdict['Ds'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Rg':
                massC.append(massdict['Rg'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Cn':
                massC.append(massdict['Cn'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Nh':
                massC.append(massdict['Nh'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Fl':
                massC.append(massdict['Fl'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Mc':
                massC.append(massdict['Mc'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Lv':
                massC.append(massdict['Lv'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Ts':
                massC.append(massdict['Ts'])
        elif len(labelsC) > 1 and i[1].islower():
            if i[:2] == 'Og':
                massC.append(massdict['Og'])

    xyzmatA = np.array(xyzA)
    xyzmatB = np.array(xyzB)
    xyzmatC = np.array(xyzC)

    comA = getCOM(xyzmatA, massA)
    comB = getCOM(xyzmatB, massB)
    comC = getCOM(xyzmatC, massC)

    # center monomer A COM at origin:
    centered_monomer_A = xyzmatA - comA

    # center monomer B COM at origin:
    centered_monomer_B = xyzmatB - comB

    # center monomer C COM at origin:
    centered_monomer_C = xyzmatC - comC

    # compute inertia tensor
    inertia_tensor_A = compute_inertia_tensor(centered_monomer_A, massA)
    inertia_tensor_B = compute_inertia_tensor(centered_monomer_B, massB)
    inertia_tensor_C = compute_inertia_tensor(centered_monomer_C, massC)

    # compute eigenvalues and eigenvectors
    # (eigenvalues are sorted in ascending order)
    eigenvalues_A, eigenvectors_A = np.linalg.eigh(inertia_tensor_A)
    eigenvalues_B, eigenvectors_B = np.linalg.eigh(inertia_tensor_B)
    eigenvalues_C, eigenvectors_C = np.linalg.eigh(inertia_tensor_C)

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    #$$$$$$$$$$$$$$$$$$$$$$$$$$ ROTATE MONOMER A #$$$$$$$$$$$$$$$$$$$$$$$$$$

    principal_axes_A = eigenvectors_A

    # transform monomer A coordinates to principal axes frame
    centered_monomer_A_inPA = np.dot(centered_monomer_A, principal_axes_A.T)

    # construct rotation matrix for monomer A
    rotation_matA = euler_rotation_matrix(alphaA, betaA, gammaA)

    # apply the rotation matrix in the principal axes frame
    centered_monomer_A_rotated_inPA = np.dot(centered_monomer_A_inPA, rotation_matA)

    # transform back to original frame
    centered_monA_rotated = np.dot(centered_monomer_A_rotated_inPA, principal_axes_A)

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    #$$$$$$$$$$$$$$$$$$$$$$$$$$ ROTATE MONOMER B #$$$$$$$$$$$$$$$$$$$$$$$$$$

    principal_axes_B = eigenvectors_B

    # transform monomer B coordinates to principal axes frame
    centered_monomer_B_inPA = np.dot(centered_monomer_B, principal_axes_B.T)

    rotation_matB = euler_rotation_matrix(alphaB, betaB, gammaB)

    # apply the rotation matrix in the principal axes frame
    centered_monomer_B_rotated_inPA = np.dot(centered_monomer_B_inPA, rotation_matB)

    # transform back to original frame
    centered_monB_rotated = np.dot(centered_monomer_B_rotated_inPA, principal_axes_B)

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    #$$$$$$$$$$$$$$$$$$$$$$$$$$ ROTATE MONOMER C #$$$$$$$$$$$$$$$$$$$$$$$$$$

    principal_axes_C = eigenvectors_C

    # transform monomer C coordinates to principal axes frame
    centered_monomer_C_inPA = np.dot(centered_monomer_C, principal_axes_C.T)

    rotation_matC = euler_rotation_matrix(alphaC, betaC, gammaC)

    # apply the rotation matrix in the principal axes frame
    centered_monomer_C_rotated_inPA = np.dot(centered_monomer_C_inPA, rotation_matC)

    # transform back to original frame
    centered_monC_rotated = np.dot(centered_monomer_C_rotated_inPA, principal_axes_C)

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    #$$$$$$$$$$$$$$$$$$$$$$$$ PERFORM TRANSLATIONS #$$$$$$$$$$$$$$$$$$$$$$$$

    # calculate the angle between AB and AC using the law of cosines
    cos_theta = (sepAB**2 + sepAC**2 - sepBC**2) / (2 * sepAB * sepAC)
    with np.errstate(invalid='ignore'):
        theta = np.arccos(cos_theta)

    translation_B = sepAB * np.array([0, 0, 1])
    translation_C = np.array([sepAC * np.sin(theta), 0, sepAC * np.cos(theta)])

    # translate molecule B 
    trans_rot_monomer_B = centered_monB_rotated + translation_B

    # translate molecule C 
    trans_rot_monomer_C = centered_monC_rotated + translation_C

    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$

    namesA = np.array(labelsA)
    namesB = np.array(labelsB)
    namesC = np.array(labelsC)

    fmonA = np.array([[f"{num:.6f}" for num in row] for row in centered_monA_rotated.astype(float)])
    fmonB = np.array([[f"{num:.6f}" for num in row] for row in trans_rot_monomer_B.astype(float)])
    fmonC = np.array([[f"{num:.6f}" for num in row] for row in trans_rot_monomer_C.astype(float)])
    monomerA = np.insert(fmonA.astype(str), 0, namesA, axis=1)
    monomerB = np.insert(fmonB.astype(str), 0, namesB, axis=1)
    monomerC = np.insert(fmonC.astype(str), 0, namesC, axis=1)

    return monomerA, monomerB, monomerC
