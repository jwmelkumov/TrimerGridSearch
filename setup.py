from setuptools import setup, find_packages

setup(
    name='TrimerGridSearch',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0', 
        'pandas>=1.3.0', 
    ],
    entry_points={
        'console_scripts': [
            'trimergridsearch = TrimerGridSearch.TrimerGridSearch:main'
        ],
    },
    description="""
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
""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jwmelkumov/TrimerGridSearch',  
    author='John W. Melkumov',
    author_email='melkumov@udel.edu',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
    ],
    python_requires='>=3.6',
)

