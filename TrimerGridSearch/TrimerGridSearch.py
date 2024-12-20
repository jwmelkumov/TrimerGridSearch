#!/usr/bin/env python3
import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import math
import sys
import os
import csv
from .guiding_potential import *
from .pyTrimerGen import TrimerGen
from .detect_clashes import detect_clashes
"""
  ============================================================================
  TrimerGridSearch:
  ============================================================================
  This package utilizes differential evolution to explore the 11-dimensional 
  potential energy surface (PES) of a trimer, starting from isolated monomers, 
  in order to identify its global minimum. The PES is evaluated at each grid 
  point using a set of force field parameters (supplied by the user) to 
  compute the two-body contribution to the trimer interaction energy.
  After the global minimum trimer has been identified, the program can be used 
  to generate a set of trimers distributed about the minimum, by perturbing
  the geometric parameters that describe the minimum trimer configuration 
  (i.e., the center-of-mass (COM) separations between monomers and Euler 
  angles- see https://github.com/jwmelkumov/TrimerGen for more info). 
  Nonsensical configurations (e.g., those with atomic clashes) are 
  automatically detected and discarded. This program currently supports 
  Lennard-Jones 12-6 + Coulomb potentials but can easily be extended to 
  support custom potentials.
  ============================================================================
  Author: John W. Melkumov
  Date: 10.08.2024.
  ============================================================================
"""
script_base_path = os.path.dirname(__file__)

def run_trimer_generator(xyzfileA, xyzfileB, xyzfileC, params):
    """
    Generates a combined np.array of three monomers using the TrimerGen function.

    Parameters:
        xyzfileA (str): Monomer A XYZ file.
        xyzfileB (str): Monomer B XYZ file.
        xyzfileC (str): Monomer C XYZ file.
        params (list/tuple): List or tuple containing 11 parameters:
            [RCOMAB, RCOMBC, RCOMAC, betaA, gammaA, alphaB, betaB, gammaB, alphaC, betaC, gammaC]

    Returns:
        np.array: Combined array containing all three monomers.
    """
    # Call TrimerGen with the given parameters
    monomerA, monomerB, monomerC = TrimerGen(
        xyzfileA, xyzfileB, xyzfileC,
        f"{params[0]:.3f}", f"{params[1]:.3f}", f"{params[2]:.3f}",  # RCOMAB, RCOMBC, RCOMAC
        f"{params[3]:.0f}", f"{params[4]:.0f}",  # betaA, gammaA
        f"{params[5]:.0f}", f"{params[6]:.0f}", f"{params[7]:.0f}",  # alphaB, betaB, gammaB
        f"{params[8]:.0f}", f"{params[9]:.0f}", f"{params[10]:.0f}"  # alphaC, betaC, gammaC
    )

    def process_monomer(monomer):
        """
        Ensures atom names are strings and coordinates are floats.
        """
        atom_names = monomer[:, 0]  # Extract atom names 
        coordinates = monomer[:, 1:].astype(float)  # Convert coordinates to float
        return np.column_stack((atom_names, coordinates))  # Recombine atom names with coordinates

    # Process each monomer
    monomerA = process_monomer(monomerA)
    monomerB = process_monomer(monomerB)
    monomerC = process_monomer(monomerC)

    # Combine all three monomers into a single array
    trimer = np.vstack((monomerA, monomerB, monomerC))

    return trimer

def write_trimer_xyz(trimer, i):
    """
    Writes a given trimer to an XYZ file.

    Parameters:
        trimer (np.array): Numpy array containing atom labels and coordinates for the trimer.
        i (int/str): Label for the output file (e.g., trimer_1.xyz).
    """
    filename = f"trimer_{i}.xyz"
    
    with open(filename, 'w') as f:
        # Write the first line: total number of atoms (rows in trimer)
        num_atoms = len(trimer)
        f.write(f"{num_atoms}\n")
        
        # Write the second line: header with "trimer_i"
        f.write(f"trimer_{i}\n")
        
        # Write the atom data (label + coordinates for each atom in trimer)
        for row in trimer:
            atom_label = str(row[0])  # Atom label
            x, y, z = float(row[1]), float(row[2]), float(row[3])  # Coordinates (x, y, z)
            f.write(f"{atom_label} {x:.6f} {y:.6f} {z:.6f}\n")

def detect_nans(trimer):
    """
    Detects NaN values in trimer numpy array.

    Parameters:
        trimer (np.array): Combined array of three monomers.

    Returns:
        bool: True if NaN values are detected, False otherwise.
    """
    for row in trimer:
        for value in row[1:]: 
            if value == 'nan' or (isinstance(value, float) and np.isnan(value)):
                return True
    return False
    
def compute_e2int_trimer(trimer, numA, numB, numC):
    """
    Computes the two-body contribution to the trimer interaction energy 
    for a given trimer.

    Parameters:
        trimer (np.array): Combined array of three monomers with atomic labels and coordinates.
        numA (int): Number of atoms in monomer A.
        numB (int): Number of atoms in monomer B.
        numC (int): Number of atoms in monomer C.

    Returns:
        float: Two-body contribution to the trimer interaction energy (e2int).
    """
    ffparams, atom_types = read_ff_params()
    
    # Extract monomer info from the trimer array
    monomerA = trimer[:numA, :]  
    monomerB = trimer[numA:numA+numB, :]  
    monomerC = trimer[numA+numB:numA+numB+numC, :]  
    
    # Compute e2int using monomer and trimer info
    e2int = compute_e2int(ffparams, atom_types, monomerA, monomerB, monomerC)
    
    return e2int

def wrap_angle(angle):
    # Wrap an angle to be within the range of [-360, 360]
    return (angle + 360) % 720 - 360

# Find global minimum trimer using differential evolution
def find_min_de(xyzfileA, xyzfileB, xyzfileC, numA, numB, numC, minrcom=2.0, maxrcom=12.0, de_pop_size=50, de_max_iter=1000, de_tol=1e-4):
    def objective_function(params):
        rAB, rBC, rAC, betaA, gammaA, alphaB, betaB, gammaB, alphaC, betaC, gammaC = params
        trimer = run_trimer_generator(xyzfileA, xyzfileB, xyzfileC, params)
        e2int = compute_e2int_trimer(trimer, numA, numB, numC)
        return e2int if not np.isnan(e2int) else np.inf  # Penalize bad trimers

    # Bounds for parameters
    bounds = [
        (minrcom, maxrcom),       # RCOMAB
        (minrcom, maxrcom),       # RCOMBC
        (minrcom, maxrcom),       # RCOMAC
        (0.0, 180.0),             # betaA
        (-360.0, 360.0),          # gammaA
        (-360.0, 360.0),          # alphaB
        (0.0, 180.0),             # betaB
        (-360.0, 360.0),          # gammaB
        (-360.0, 360.0),          # alphaC
        (0.0, 180.0),             # betaC
        (-360.0, 360.0),          # gammaC
    ]

    # Perform global optimization
    print("Starting global search using Differential Evolution...")
    result_de = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=de_max_iter,
        popsize=de_pop_size,
        tol=de_tol,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True
    )

    if result_de.success:
        print("Global search successfully converged.")
        print("*******************************************************************")
        print(f"Global minimum trimer found:")
        print("*******************************************************************")
        optimized_params = result_de.x
        print(f"RCOMAB: {optimized_params[0]}, RCOMBC: {optimized_params[1]}, RCOMAC: {optimized_params[2]}")
        print(f"betaA: {optimized_params[3]}, gammaA: {optimized_params[4]}")
        print(f"alphaB: {optimized_params[5]}, betaB: {optimized_params[6]}, gammaB: {optimized_params[7]}")
        print(f"alphaC: {optimized_params[8]}, betaC: {optimized_params[9]}, gammaC: {optimized_params[10]}")
        print(f"Final E2int: {result_de.fun}")
        print("*******************************************************************")
    else:
        print("Global search did not converge.")
        print("*******************************************************************")
        print(f"Last evaluated function value: {result_de.fun}")
        print(f"Last set of parameters: {result_de.x}")
        print("*******************************************************************")

    return result_de.x, result_de.fun

def sample_around_min(min_params, xyzfileA, xyzfileB, xyzfileC, numA, numB, numC, num_samples=100, minrcom=2.0, maxrcom=12.0, rep_wall=100, delta=0.5):
    file_exists = os.path.isfile('temp_trimer_grid.csv')
    if file_exists:
        os.remove('temp_trimer_grid.csv')

    with open('temp_trimer_grid.csv', mode='w', newline='') as csvfile:
        fieldnames = ['RCOMAB', 'RCOMBC', 'RCOMAC', 'betaA', 'gammaA', 'alphaB', 
                      'betaB', 'gammaB', 'alphaC', 'betaC', 'gammaC', 'E2int']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    valid_samples = []  

    with open('temp_trimer_grid.csv', mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['RCOMAB', 'RCOMBC', 'RCOMAC', 'betaA', 'gammaA', 'alphaB', 
                                                     'betaB', 'gammaB', 'alphaC', 'betaC', 'gammaC', 'E2int'])

        # Loop through and generate multiple trimer samples
        for i in range(num_samples):  # Generate num_samples trimers
            success = False  # Track if we successfully generate a valid trimer

            while not success:  # Keep generating until a valid trimer is found
                # Sample about the minimum using perturbation delta
                sampled_params = []
                for j in range(len(min_params)):
                    if j < 3:  # RCOM distances
                        sampled_param = np.random.uniform(min_params[j] - delta, min_params[j] + delta)
                        sampled_param = round(np.clip(sampled_param, minrcom, maxrcom), 2)  
                    else:  # Angles (beta, gamma, alpha)
                        sampled_param = min_params[j]  
                        sampled_param = round(sampled_param, 0)
                        if j == 5 or j == 8:  # alpha angles (-360 to 360)
                            sampled_param = round(np.clip(sampled_param + np.random.uniform(-10*delta, 10*delta), -360, 360), 2)
                        elif j == 3 or j == 6 or j == 9:  # beta angles (0 to 180)
                            sampled_param = round(np.clip(sampled_param + np.random.uniform(-10*delta, 10*delta), 0, 180), 2)
                        elif j == 4 or j == 7 or j == 10:  # gamma angles (-360 to 360)
                            sampled_param = round(np.clip(sampled_param + np.random.uniform(-10*delta, 10*delta), -360, 360), 2)

                    # Append after all checks
                    sampled_params.append(sampled_param)
                
                # Generate trimer using the sampled parameters
                generatedtrimer = run_trimer_generator(xyzfileA, xyzfileB, xyzfileC, sampled_params)

                # Detect clashes and resample if necessary
                if detect_clashes(generatedtrimer, numA, numB, numC):
                    #print("Clash detected. Resampling...")
                    continue

                # Check for NaN coordinates and resample if necessary
                if detect_nans(generatedtrimer):
                    #print("NaN detected in generated trimer. Resampling...")
                    continue

                # Compute E2int for the generated trimer
                e2int = compute_e2int_trimer(generatedtrimer, numA, numB, numC)

                # Check if E2int is greater than the repulsive wall and resample if necessary
                if e2int > rep_wall:
                    #print(f"E2int ({e2int}) is greater than the repulsive wall. Resampling...")
                    continue

                # Check if E2int is NaN and resample if necessary
                if math.isnan(e2int):
                    #print("E2int is NaN. Resampling...")
                    continue

                print(f"Sampled parameters: {sampled_params}")
                print(f"E2int for sample: {e2int}")

                # Generated trimer is valid
                success = True  # Mark success

            # Write the sampled parameters and E2int to the list of valid samples
            writer.writerow({
                'RCOMAB': sampled_params[0],
                'RCOMBC': sampled_params[1],
                'RCOMAC': sampled_params[2],
                'betaA': sampled_params[3],
                'gammaA': sampled_params[4],
                'alphaB': sampled_params[5],
                'betaB': sampled_params[6],
                'gammaB': sampled_params[7],
                'alphaC': sampled_params[8],
                'betaC': sampled_params[9],
                'gammaC': sampled_params[10],
                'E2int': e2int
            })
    
    print("Finished generating intermediate trimer batch.")

def main():
    xyzfileA = sys.argv[1]
    xyzfileB = sys.argv[2]
    xyzfileC = sys.argv[3]
    numA = get_numX(xyzfileA)
    numB = get_numX(xyzfileB)
    numC = get_numX(xyzfileC)

    print(f"===========================================================")
    print(f"""
    .######..#####...######..##...##..######..#####..
    ...##....##..##....##....###.###..##......##..##.
    ...##....#####.....##....##.#.##..####....#####..
    ...##....##..##....##....##...##..##......##..##.
    ...##....##..##..######..##...##..######..##..##.
    .................................................
    ..####...#####...######..#####..                 
    .##......##..##....##....##..##.                 
    .##.###..#####.....##....##..##.                 
    .##..##..##..##....##....##..##.                 
    ..####...##..##..######..#####..                 
    ................................                 
    ..####...######...####...#####....####...##..##. 
    .##......##......##..##..##..##..##..##..##..##. 
    ..####...####....######..#####...##......######. 
    .....##..##......##..##..##..##..##..##..##..##. 
    ..####...######..##..##..##..##...####...##..##. 
    ................................................                                                                 
            """)    
    print(f"===========================================================")

    # Initialize hyperparameters with default values
    de_max_iter=1000          # Default max number of iterations of DE
    de_pop_size = 50          # Default population size for DE
    de_tol=1e-4               # Default tolerance for DE convergence 
    num_samples = 100         # Default number of samples to generate about minimum
    delta = 0.5               # Default delta for perturbation-based sampling
    rep_wall = 40             # Default repulsive wall (in kcal/mol)
    minrcom = 2.0             # Default minrcom (in Angstrom)
    maxrcom = 12.0            # Default maxrcom (in Angstrom)
    writexyzfiles = False     # Default setting for writing XYZ files of samples about minimum

    # Read hyperparameters from trimer.input:
    with open('trimer.input', 'r') as f:
        lines = f.readlines()
        for line in lines:

            line = line.strip()

            if line.startswith("de_max_iter"):
                de_max_iter = int(line.split('=')[1].strip())
                print(f"de_max_iter: {de_max_iter}")
            elif line.startswith("de_pop_size"):
                de_pop_size = int(line.split('=')[1].strip())
                print(f"de_pop_size: {de_pop_size}")
            elif line.startswith("de_tol"):
                de_tol = float(line.split('=')[1].strip())
                print(f"de_tol: {de_tol}")
            elif line.startswith("num_samples"):
                num_samples = int(line.split('=')[1].strip())
                print(f"num_samples: {num_samples}")
            elif line.startswith("delta"):
                delta = float(line.split('=')[1].strip())
                print(f"delta: {delta}")
            elif line.startswith("rep_wall"):
                rep_wall = float(line.split('=')[1].strip())
                print(f"rep_wall: {rep_wall}")
            elif line.startswith("minrcom"):
                minrcom = float(line.split('=')[1].strip())
                print(f"minrcom: {minrcom}")
            elif line.startswith("maxrcom"):
                maxrcom = float(line.split('=')[1].strip())
                print(f"maxrcom: {maxrcom}")
            elif line.startswith("writexyzfiles"):
                writexyzfiles = bool(line.split('=')[1].strip().lower())
                if writexyzfiles == 'true':
                    writexyzfiles = True
                else:
                    writexyzfiles = False
                print(f"writexyzfiles: {writexyzfiles}")
    print(f"===========================================================")

    if de_max_iter <= 0.0 or de_pop_size <= 0.0 or de_tol <= 0.0 or num_samples <= 0.0 or delta <= 0.0 or rep_wall <= 0.0 or minrcom <= 0.0 or maxrcom <= 0.0:
        print("Error: de_max_iter, de_pop_size, de_tol, num_samples, delta, rep_wall, minrcom, and maxrcom all must be greater than 0.")
        sys.exit(1)

    valid_samples = []
    
    # Find global minimum trimer
    min_params, e2intmin = find_min_de(xyzfileA, xyzfileB, xyzfileC, numA, numB, numC, de_max_iter=de_max_iter, de_pop_size=de_pop_size, de_tol=de_tol)
    mintrimer = run_trimer_generator(xyzfileA, xyzfileB, xyzfileC, min_params)
    write_trimer_xyz(mintrimer, "min")

    # Generate trimers about global minimum
    for i in range(num_samples):
        sample_around_min(min_params, xyzfileA, xyzfileB, xyzfileC, numA, numB, numC, num_samples=num_samples, minrcom=minrcom, maxrcom=maxrcom, rep_wall=rep_wall, delta=delta)
        generated_samples = pd.read_csv('temp_trimer_grid.csv')
        filtered_samples = generated_samples[
            ((generated_samples['E2int'] >= 0) & (generated_samples['E2int'] <= rep_wall)) |
            ((generated_samples['E2int'] <= 0) & (generated_samples['E2int'] >= e2intmin)) |
            (generated_samples['E2int'] < 0)
        ]
        valid_samples.append(filtered_samples)

    # Combine all valid samples into single dataframe
    all_valid_samples = pd.concat(valid_samples, ignore_index=True)

    # Save all valid samples to final CSV and remove temp file
    all_valid_samples.to_csv('final_trimer_grid.csv', index=False)
    os.remove('temp_trimer_grid.csv')

    # If requested, generate trimer xyz files
    paramcols = all_valid_samples.columns.difference(['E2int'])
    if writexyzfiles:
        for i in range(len(all_valid_samples)):
            params = all_valid_samples.iloc[i][paramcols]
            trimer = run_trimer_generator(xyzfileA, xyzfileB, xyzfileC, params)
            write_trimer_xyz(trimer, i)
    else:
        print("Trimer generation finished. Final grid data written to final_trimer_grid.csv")
    
if __name__ == "__main__":
    main()
