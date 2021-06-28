#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from NetCDFIO cimport  NetCDFIO_Stats
from TimeStepping cimport TimeStepping
from Variables cimport GridMeanVariables, VariablePrognostic
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin
import netCDF4 as nc
from scipy.interpolate import interp2d

cdef class RadiationBase:
    cdef:
        double [:] dTdt
        double [:] dqtdt

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationTRMM_LBA(RadiationBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)


cdef class RadiationDYCOMS_RF01(RadiationBase):
    cdef:
        double alpha_z
        double kappa
        double F0
        double F1
        double divergence
        double [:] f_rad # radiative flux at cell edges
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationLES(RadiationBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)