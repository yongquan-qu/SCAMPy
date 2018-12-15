# SCAMPy #

SCAMPy (Single Column Atmospheric Model in Python) provides a framework for testing parameterizations of clouds and turbulence.
It is particularly designed to support eddy-diffusivity mass-flux modeling frameworks.

Information about the EDMF parameterization implemented in SCAMPy can be found in:

Tan, Z., C. M. Kaul, K. G. Pressel, Y. Cohen, T. Schneider, and J. Teixeira, 2018:
An extended eddy-diffusivity mass-flux scheme for unified representation of
subgrid-scale turbulence and convection. Journal of Advances in Modeling Earth Systems, 2018.
(see [Tan et al., 2018](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017MS001162)).

The code is written in Python and Cython.

Code Contributors:
	Colleen Kaul (Caltech)--initial/primary developer.
	Yair Cohen (Caltech);
	Anna Jaruga (JPL/Caltech);
        Ignacio Lopez-Gomez (Caltech);
	Kyle Pressel (Caltech);
	Zhihong Tan (U. Chicago)

Additional Acknowledgements:
	Tapio Schneider (Caltech);
	Joao Teixeira (JPL)

# installation #

SCAMPy is written in Python and is compatible with both Python 2.7 and 3.6.
It requires [Cython](http://cython.org/), [netcdf4](http://unidata.github.io/netcdf4-python/)
 and [scipy](https://www.scipy.org/).
For testing and plotting it additionally requires [pytest](https://docs.pytest.org/en/latest/)
 and [matplotlib](https://matplotlib.org/).
All the dependancies can be installed using [pip](https://pypi.org/project/pip/).
See also the travis.yml file for a complete list of instructions needed to compile and run SCAMPy on a clean Linux machine.

# building and running #
```
$ cd scampy
```

Generate the simulation specific parameters (accepted keywords: Soares, DYCOMS_RF01, DYCOMS_RF02, Bomex, life_cycle_Tan2018, Rico, TRMM_LBA, ARM_SGP, GATE_III)
```
$ python generate_namelist.py Soares
```

Generate the turbulence parameters (accepted keywords: defaults, Soares, DYCOMS_RF01, DYCOMS_RF02, Bomex, life_cycle_Tan2018, Rico, TRMM_LBA, ARM_SGP, GATE_III)
```
$ python generate_paramlist.py Soares
```

Compile the source code by running setup.py
```
$ python setup.py build_ext --inplace
```

Serial execution of scampy (both turbulence and case specific parameters need to be passed)
```
$ python main.py Soares.in paramlist_Soares.in
```

# testing  #

Tests for SCAMPy are located in the tests folder.

To run automatic plots please try:

```
$ cd tests/

$ py.test -v plots/

$ cd ../

```

TODO: regression tests

TODO: unit tests
