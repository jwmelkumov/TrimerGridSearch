# TrimerGridSearch
This package utilizes differential evolution to explore the 11-dimensional 
potential energy surface (PES) of a trimer, starting from isolated monomers, 
in order to identify its global minimum. The PES is evaluated at each grid 
point using a set of force field parameters (supplied by the user) to 
compute the two-body contribution to the trimer interaction energy.
After the global minimum trimer has been identified, the program can be used 
to generate a set of trimers distributed about the minimum, by perturbing
the geometric parameters that describe the minimum trimer configuration 
(i.e., the center-of-mass (COM) separations between monomers and Euler 
angles- see [TrimerGen](https://github.com/jwmelkumov/TrimerGen for more info)). 
Nonsensical configurations (e.g., those with atomic clashes) are 
automatically detected and discarded. This program currently supports 
Lennard-Jones 12-6 + Coulomb potentials but can easily be extended to 
support custom potentials.

### Dependencies
- Numpy
- Subprocess
- Python3

## Install
```bash
pip install TrimerGridSearch
```

## Necessary Input Files

### trimer.input
Contains hyperparameter information related to run (e.g., de_max_iter, de_tol).

## trimer.fftop
Contains mapping information used to associate atom labels with atom types.

## trimer.ffprm
Contains atom types and their associated force field parameters (i.e., charge,
sigma, epsilon).

## Usage
```bash
trimergridsearch <monomerA.xyz> <monomerB.xyz> <monomerC.xyz> 
```

## Example
```bash
trimergridsearch meoh.xyz meoh.xyz meoh.xyz 
```
