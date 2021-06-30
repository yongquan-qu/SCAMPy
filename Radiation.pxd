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
        Grid Gr
        double (*convert_forcing_prog_fp)(double p0, double qt, double qv, double T,
                                          double qt_tendency, double T_tendency) nogil
        ReferenceState Ref

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationTRMM_LBA(RadiationBase):
    cdef:
        double [:] rad_time
        double [:,:] rad

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
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
    cpdef calculate_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class RadiationLES(RadiationBase):
    cdef:
        double [:,:] dtdt_rad
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)