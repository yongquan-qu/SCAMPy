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

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingStandard(ForcingBase):
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingDYCOMS_RF01(ForcingBase):
    cdef:
        double alpha_z
        double kappa
        double F0
        double F1
        double divergence
        double [:] f_rad # radiative flux at cell edges

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingLES(ForcingBase):
    cdef:
        str LES_filename
        double nudge_tau
        double [:] t_les
        double [:] z_les
        double [:,:] scm_subsidence
        double [:,:] dtdt_rad
        double [:,:] dtdt_hadv
        double [:,:] dtdt_nudge
        double [:,:] dqtdt_hadv
        double [:,:] dqtdt_nudge
        double [:,:] u_nudge
        double [:,:] v_nudge

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
