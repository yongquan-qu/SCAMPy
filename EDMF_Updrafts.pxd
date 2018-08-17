cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from EDMF_Rain cimport RainVariables

cdef class UpdraftVariable:
    cdef:
        double [:,:] values
        double [:,:] new
        double [:,:] old
        double [:,:] tendencies
        double [:,:] flux
        double [:] bulkvalues
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self, Grid.Grid Gr)

cdef class UpdraftVariables:
    cdef:
        Grid.Grid Gr
        UpdraftVariable W
        UpdraftVariable Area
        UpdraftVariable QT
        UpdraftVariable QL
        UpdraftVariable H
        UpdraftVariable THL
        UpdraftVariable T
        UpdraftVariable B
        Py_ssize_t n_updrafts
        bint prognostic
        double updraft_fraction
        double [:] cloud_base
        double [:] cloud_top
        double [:] updraft_top
        double [:] cloud_cover
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef set_means(self, GridMeanVariables GMV)
    cpdef set_new_with_values(self)
    cpdef set_old_with_values(self)
    cpdef set_values_with_new(self)
    cpdef get_cloud_base_top_cover(self)

cdef class UpdraftThermodynamics:
    cdef:
        double (*t_to_prog_fp)(double p0, double T, double qt, double ql, double qi) nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil

        Grid.Grid Gr
        ReferenceState.ReferenceState Ref
        Py_ssize_t n_updraft

        double [:,:] prec_source_h
        double [:,:] prec_source_qt
        double [:] prec_source_h_tot
        double [:] prec_source_qt_tot

    cpdef buoyancy(
        self, UpdraftVariables UpdVar, EnvironmentVariables EnvVar,
        GridMeanVariables GMV, bint extrap
    )

    # helper functions to calculate autoconversion source terms to THL and QT
    cpdef clear_precip_sources(self)
    cpdef update_total_precip_sources(self)

    # update all variables (for ! use_local_micro)
    cpdef update_column_UpdVar_UpdRain(
        self, UpdraftVariables UpdVar, RainVariables Rain
    )

    # update variables level by level (for use_local_micro)
    cdef void update_UpdVar(
        self, double *qt, double *ql, double *h, double *T, double *area,
        double qr_src, double th_src, double qt_new, double ql_new,
        double T_new, double thl_new, Py_ssize_t i, Py_ssize_t k
    ) nogil
    cdef void update_UpdRain(
        self, double *upd_area, double *qr, double *rain_area, double qr_new,
        double a_const, Py_ssize_t i, Py_ssize_t k
    ) nogil
