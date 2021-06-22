from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables, VariablePrognostic
from NetCDFIO cimport  NetCDFIO_Stats
from TimeStepping cimport TimeStepping

cdef class ForcingBase:
    cdef:
        double [:] subsidence
        double [:] dTdt # horizontal advection temperature tendency
        double [:] dqtdt # horizontal advection moisture tendency
        bint apply_coriolis
        bint apply_subsidence
        double coriolis_param
        double [:] ug
        double [:] vg

        double (*convert_forcing_prog_fp)(double p0, double qt, double qv, double T,
                                          double qt_tendency, double T_tendency) nogil
        Grid Gr
        ReferenceState Ref

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS, namelist)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS,namelist)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingStandard(ForcingBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS,namelist)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

# cdef class ForcingRadiative(ForcingBase):
#     cpdef initialize(self, GridMeanVariables GMV)
#     cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
#     cpdef initialize_io(self, NetCDFIO_Stats Stats)
#     cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingDYCOMS_RF01(ForcingBase):
    cdef:
        double alpha_z
        double kappa
        double F0
        double F1
        double divergence
        double [:] f_rad # radiative flux at cell edges

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS,namelist)
    cpdef calculate_radiation(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingLES(ForcingBase):
    cdef:
        str LES_filename
        double [:] t_les # radiative dTdt from les
        double [:] z_les # radiative dTdt from les
        double [:,:] les_dtdt_rad # radiative dTdt from les
        double [:,:] les_dtdt_hadv # radiative dTdt from les
        double [:,:] les_dtdt_nudge # radiative dTdt from les
        double [:,:] les_dqtdt_rad # radiative dTdt from les
        double [:,:] les_dqtdt_hadv # radiative dTdt from les
        double [:,:] les_dqtdt_nudge # radiative dTdt from les
        double [:,:] les_subsidence # radiative dTdt from les
        double [:,:] scm_subsidence # radiative dTdt from les
        double [:,:] dtdt_hadv
        double [:,:] dtdt_nudge
        double [:,:] dqtdt_rad
        double [:,:] dqtdt_hadv
        double [:,:] dqtdt_nudge
        double [:,:] dtdt_rad # radiative dTdt from les

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS,namelist)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
