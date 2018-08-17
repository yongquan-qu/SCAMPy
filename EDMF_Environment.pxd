from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from EDMF_Rain cimport RainVariables

cdef class EnvironmentVariable:
    cdef:
        double [:] values
        double [:] flux
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self,Grid Gr)

cdef class EnvironmentVariable_2m:
    cdef:
        double [:] values
        double [:] dissipation
        double [:] shear
        double [:] entr_gain
        double [:] detr_loss
        double [:] press
        double [:] buoy
        double [:] interdomain
        double [:] rain_src
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self,Grid Gr)

cdef class EnvironmentVariables:
    cdef:
        EnvironmentVariable W
        EnvironmentVariable QT
        EnvironmentVariable QL
        EnvironmentVariable EnvArea
        EnvironmentVariable H
        EnvironmentVariable THL
        EnvironmentVariable T
        EnvironmentVariable B
        EnvironmentVariable_2m TKE
        EnvironmentVariable_2m Hvar
        EnvironmentVariable_2m QTvar
        EnvironmentVariable_2m HQTcov
        EnvironmentVariable CF
        Grid Gr

        bint calc_tke
        bint calc_scalar_var
        bint use_quadrature

        bint use_quadrature

        str EnvThermo_scheme

    cpdef initialize_io(self, NetCDFIO_Stats Stats )
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class EnvironmentThermodynamics:
    cdef:
        Grid Gr
        ReferenceState Ref
        Py_ssize_t quadrature_order

        double (*t_to_prog_fp)(double p0, double T, double qt, double ql, double qi) nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil

        double [:] qt_dry
        double [:] th_dry
        double [:] t_cloudy
        double [:] qv_cloudy
        double [:] qt_cloudy
        double [:] th_cloudy

        double [:] Hvar_rain_dt
        double [:] QTvar_rain_dt
        double [:] HQTcov_rain_dt

        void update_EnvVar(self,    Py_ssize_t k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double alpha) nogil
        void update_EnvRain(self,   Py_ssize_t k, EnvironmentVariables EnvVar, RainVariables Rain, double qr) nogil
        void update_cloud_dry(self, Py_ssize_t k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double qv) nogil

        void eos_update_SA_smpl(self, EnvironmentVariables EnvVar)
        void eos_update_SA_mean(self, EnvironmentVariables EnvVar, RainVariables Rain)
        void eos_update_SA_sgs(self,  EnvironmentVariables EnvVar, RainVariables Rain)

    cpdef satadjust(self, EnvironmentVariables EnvVar, RainVariables Rain)
