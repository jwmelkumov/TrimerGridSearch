#!/usr/bin/env python3
import sys
import os
import numpy as np
from .detect_clashes import distance
"""
  ============================================================================
  Force Field Utility Module
  ============================================================================
  Author: John W. Melkumov
  Date: 10.02.2024.
  ============================================================================
"""
def read_ff_params():
    """Reads file containing info tying an atom name label 
       in the trimer to a force field atom type and then uses that atom 
       type to look up parameters from a force field parameter file
       (parfile).
    """
    topfile = "trimer.fftop"
    parfile = "trimer.ffprm"
    with open(topfile, 'r') as f:
        lines = f.readlines()
    ffparams = {}

    numA, numB, numC = map(int, lines[0].split())
    
    atom_types = {}
    
    # Read in the atom names and their force field atom types
    for line in lines[1:]:
        atom_name, atom_type = line.split()
        atom_types[atom_name] = int(atom_type)

    # Look up the force field parameters for each atom type
    with open(parfile, 'r') as f:
        next(f)
        for line in f:
            atom_type, charge, sigma, epsilon = line.split()
            ffparams[atom_type] = {'charge': float(charge), 'sigma': float(sigma), 'epsilon': float(epsilon)}
    
    return ffparams, atom_types

def compute_e2int(ffparams, atom_types, monomerA, monomerB, monomerC):
    """Uses force field parameters and coordinates of passed in trimer 
       geometry to compute 2-body contribution to trimer interaction
       energy.
    """
    e2int = 0.0
    eintab, eintbc, eintac = 0.0, 0.0, 0.0
    
    pairsab = []
    pairsbc = []
    pairsac = []

    # Collect pairs for AB
    for i in range(len(monomerA)):
        for j in range(len(monomerB)):
            pairsab.append((monomerA[i, 0:], monomerB[j, 0:]))

    # Collect pairs for BC
    for j in range(len(monomerB)):
        for k in range(len(monomerC)): 
            pairsbc.append((monomerB[j, 0:], monomerC[k, 0:]))

    # Collect pairs for AC
    for i in range(len(monomerA)):
        for k in range(len(monomerC)):
            pairsac.append((monomerA[i, 0:], monomerC[k, 0:]))

    # Compute the interaction energies:
    # AB pairs
    for a, b in pairsab:
        r = distance(a, b)
        q1 = ffparams[str(atom_types[a[0]])]['charge']
        q2 = ffparams[str(atom_types[b[0]])]['charge']
        epsilon1 = ffparams[str(atom_types[a[0]])]['epsilon']
        epsilon2 = ffparams[str(atom_types[b[0]])]['epsilon']
        sigma1 = ffparams[str(atom_types[a[0]])]['sigma']
        sigma2 = ffparams[str(atom_types[b[0]])]['sigma']
        eintab += ljcoulombvij(r, q1, q2, epsilon1, epsilon2, sigma1, sigma2)

    # BC pairs
    for b, c in pairsbc:
        r = distance(b, c)
        q1 = ffparams[str(atom_types[b[0]])]['charge']
        q2 = ffparams[str(atom_types[c[0]])]['charge']
        epsilon1 = ffparams[str(atom_types[b[0]])]['epsilon']
        epsilon2 = ffparams[str(atom_types[c[0]])]['epsilon']
        sigma1 = ffparams[str(atom_types[b[0]])]['sigma']
        sigma2 = ffparams[str(atom_types[c[0]])]['sigma']
        eintbc += ljcoulombvij(r, q1, q2, epsilon1, epsilon2, sigma1, sigma2)

    # AC pairs
    for a, c in pairsac:
        r = distance(a, c)
        q1 = ffparams[str(atom_types[a[0]])]['charge']
        q2 = ffparams[str(atom_types[c[0]])]['charge']
        epsilon1 = ffparams[str(atom_types[a[0]])]['epsilon']
        epsilon2 = ffparams[str(atom_types[c[0]])]['epsilon']
        sigma1 = ffparams[str(atom_types[a[0]])]['sigma']
        sigma2 = ffparams[str(atom_types[c[0]])]['sigma']
        eintac += ljcoulombvij(r, q1, q2, epsilon1, epsilon2, sigma1, sigma2)

    e2int = (eintab + eintac + eintbc)
    return e2int

def ljcoulombvij(r, q1, q2, epsilon1, epsilon2, sigma1, sigma2):
    """Lennard-Jones 12-6 + Coulomb potential."""
    kc = 332 # Coulomb constant 1/(4*pi*e0) in kcal-A/e^2
    sigma = (sigma1 + sigma2)/2
    epsilon = np.sqrt(epsilon1 * epsilon2)
    return (kc*(q1*q2/r)) + (4 * epsilon * ((sigma/r)**12 - (sigma/r)**6))

def get_numX(file_path):
    """Reads monomer XYZ files and extracts the number of atoms."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    numX = int(lines[0].strip())
    return numX

def read_trimer_xyz(file_path):
    """Reads a given trimer XYZ file and extracts atom labels and coordinates."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(lines[0].strip())

    atoms = []
    for line in lines[2:2 + num_atoms]:  
        parts = line.strip().split()
        atom_label = parts[0]
        x, y, z = map(float, parts[1:])  
        atoms.append((atom_label, np.array([x, y, z])))
    
    return atoms

