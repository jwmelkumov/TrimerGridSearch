#!/usr/bin/env python3
import numpy as np
import sys
import os
"""
  ============================================================================
  Trimer Clash Detection Utility Module
  ============================================================================
  Author: John W. Melkumov
  Date: 10.02.2024.
  ============================================================================
"""
def distance(atom1, atom2):
    """Calculates Euclidean distance between two atoms."""
    # Assuming atom1 and atom2 are tuples (atom_name, x, y, z)
    pos1 = np.array(atom1[1:], dtype=float)  # Coordinates should be in positions 1, 2, 3
    pos2 = np.array(atom2[1:], dtype=float)
    return np.linalg.norm(pos1 - pos2)

def detect_clashes(trimer, numA, numB, numC, clash_threshold=2.0):
    """
    Detects atomic clashes within a trimer system.

    Parameters:
        trimer (np.array): Combined array of three monomers from `run_trimer_generator`.
        numA (int): Number of atoms in monomer A.
        numB (int): Number of atoms in monomer B.
        numC (int): Number of atoms in monomer C.
        clash_threshold (float): Threshold for detecting clashes in Å.

    Returns:
        bool: False if no clashes are detected, True otherwise.
    """
    # Split trimer into monomers based on atom counts
    atomsA = trimer[:numA]
    atomsB = trimer[numA:numA + numB]
    atomsC = trimer[numA + numB:numA + numB + numC]
    
#    def distance(atom1, atom2):
#        """Calculates Euclidean distance between two atoms."""
#        pos1 = np.array(atom1[1:], dtype=float)
#        pos2 = np.array(atom2[1:], dtype=float)
#        return np.linalg.norm(pos1 - pos2)

    # Check A-B clashes
    for atomA in atomsA:
        for atomB in atomsB:
            if distance(atomA, atomB) < clash_threshold:
                return True
    
    # Check A-C clashes
    for atomA in atomsA:
        for atomC in atomsC:
            if distance(atomA, atomC) < clash_threshold:
                return True
    
    # Check B-C clashes
    for atomB in atomsB:
        for atomC in atomsC:
            if distance(atomB, atomC) < clash_threshold:
                return True
    
    return False
