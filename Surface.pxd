from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from thermodynamic_functions cimport latent_heat,cpm_c
from TimeStepping cimport TimeStepping

cdef class SurfaceBase:
    cdef:
        double zrough
        bint interactive_zrough
        double Tsurface
        double qsurface
        double shf
        double lhf
        double cm
        double ch
        double cq
        double bflux
        double windspeed
        double ustar
        double rho_qtflux
        double rho_hflux
        double rho_uflux
        double rho_vflux
        double obukhov_length
        double Ri_bulk_crit
        double [:] scm_shf
        double [:] scm_lhf
        bint ustar_fixed
        Grid Gr
        ReferenceState Ref
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceNone(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceFixedFlux(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceFixedCoeffs(SurfaceBase):
    cdef:
        double s_surface
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceMoninObukhov(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceMoninObukhovDry(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

# Not fully implemented (maybe not needed) in .pyx - Ignacio
cdef class SurfaceSullivanPatton(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceLES(SurfaceBase):
    cpdef initialize(self, Grid Gr, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)
