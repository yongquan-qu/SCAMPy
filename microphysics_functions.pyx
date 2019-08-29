import numpy as np
cimport numpy as np

from libc.math cimport fmax, exp

from thermodynamic_functions cimport *
include "parameters.pxi"

cdef double r2q(double r_, double qt) nogil :
    """
    Convert mixing ratio to specific humidity assuming
    qd = 1 - qt
    qt = qv + ql + qi
    qr = mr/md+mv+ml+mi
    """
    return r_ * (1. - qt)

cdef double q2r(double q_, double qt) nogil :
    """
    Convert specific humidity to mixing ratio
    see r2q for assumptions
    """
    return q_ / (1. - qt)

cdef double rain_source_to_thetal(double p0, double T, double qr) nogil :
    """
    Source term for thetal because of qr transitioning between the working fluid and rain
    (simple version to avoid exponents)
    """
    return latent_heat(T) * qr / exner_c(p0) / cpd

cdef double rain_source_to_thetal_detailed(double p0, double T, double qt, double ql, double qr) nogil :
    """
    Source term for thetal because of qr transitioning between the working fluid and rain
    (more detailed version, but still ignoring dqt/dqr)
    """
    cdef double L = latent_heat(T)

    old_source = L * qr / exner_c(p0) / cpd

    new_source = old_source / (1.-qt) * exp(-L * ql / T / cpd / (1.-qt))

    return new_source

# instantly convert all cloud water exceeding a threshold to rain water
# the threshold is specified as axcess saturation
# rain water is immediately removed from the domain
cdef double acnv_instant(double ql, double qt, double sat_treshold, double T, double p0, double ar) nogil :

    cdef double psat = pv_star(T)
    cdef double qsat = qv_star_c(p0, qt, psat)

    if ar <= 0.:
        _ret = 0.
    else:
        _ret = fmax(0.0, ql - sat_treshold * qsat)
    return _ret

# time-rate expressions for 1-moment microphysics
# autoconversion:   Kessler 1969, see Table 1 in Wood 2005: https://doi.org/10.1175/JAS3530.1
# accretion, rain evaporation rain terminal velocity:
#    Grabowski and Smolarkiewicz 1996 eqs: 5b-5d
#    https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2

# rate expressions in the paper are for mixing ratios
# need to convert to specific humidities

cdef double acnv_rate(double ql, double qt) nogil :

    cdef double rl = q2r(ql, qt)
    cdef double  _ret

    return (1. - qt) * 1e-3 * fmax(0.0, rl - 5e-4)

cdef double accr_rate(double ql, double qr, double qt) nogil :

    cdef double rl = q2r(ql, qt)
    cdef double rr = q2r(qr, qt)

    return (1. - qt) * 2.2 * rl * rr**0.875
    #      dq/dr     * dr/dt

cdef double evap_rate(double rho, double qv, double qr, double qt, double T, double p0) nogil :

    cdef double psat = pv_star(T)
    cdef double qsat = qv_star_c(p0, qt, psat)
    cdef double rr   = q2r(qr, qt)
    cdef double rv   = q2r(qv, qt)
    cdef double rsat = q2r(qsat, qt)

    cdef double C = 1.6 + 124.9 * (1e-3 * rho * rr)**0.2046 # ventilation factor

    return (1 - qt) * (1. - rv/rsat) * C * (1e-3 * rho * rr)**0.525 / rho / (540 + 2.55 * 1e5 / (p0 * rsat))
    #      dq/dr     * dr/dt

cdef double terminal_velocity(double rho, double rho0, double qr, double qt) nogil :

    cdef double rr = q2r(qr, qt)

    return 14.34 * rho0**0.5 * rho**-0.3654 * rr**0.1346

cdef mph_struct microphysics_rain_src(double T, double ql, double p0, double qt, double area,\
                         double max_supersat) nogil:
    """
    do autoconversion
    return updated T, THL, qt, qv, ql, qr, alpha
    """
    # TODO assumes no ice
    cdef mph_struct _ret

    _ret.qv    = qt - ql
    _ret.thl   = t_to_thetali_c(p0, T, qt, ql, 0.0)
    _ret.th    = theta_c(p0, T)
    _ret.alpha = alpha_c(p0, T, qt, _ret.qv)

    _ret.qr_src       = acnv_instant(ql, qt, max_supersat, T, p0, area)
    _ret.thl_rain_src = rain_source_to_thetal(p0, T, _ret.qr_src)

    _ret.qt  = qt - _ret.qr_src
    _ret.ql  = ql - _ret.qr_src

    _ret.thl += _ret.thl_rain_src

    # new values
    #mph.qt
    #mph.ql
    #mph.qv
    #mph.thl
    #mph.th
    #sa.T
    #mph.alpha

    # rates
    #mph.qr
    #mph.thl_rain_src

    return _ret

cdef rain_struct rain_area(double source_area,  double source_qr,
                           double current_area, double current_qr ) nogil:
    """
    Source terams for rain and rain area
    assuming constant rain area fraction of 1
    """
    cdef rain_struct _ret

    if source_qr <= 0.:
        _ret.qr = current_qr
        _ret.ar = current_area
    else:
        _ret.qr = current_qr + source_area * source_qr
        _ret.ar = 1.

    # sketch of what to do for prognostic rain area fraction:

    #cdef double a_big, q_big, a_sml, q_sml
    #cdef double a_const = 0.2
    #cdef double eps     = 1e-5

    #if source_qr ==  0.:
    #    _ret.qr = current_qr
    #    _ret.ar = current_area
    #else:
    #    if current_area != 0.:
    #        if current_area >= source_area:
    #            a_big = current_area
    #            q_big = current_qr
    #            a_sml = source_area
    #            q_sml = source_qr
    #        else:
    #            a_sml = current_area
    #            q_sml = current_qr
    #            a_big = source_area
    #            q_big = source_qr

    #        _ret.qr = q_big + a_sml / a_big * q_sml
    #        _ret.ar = a_big

    #    else:
    #        _ret.qr = source_qr
    #        _ret.ar = source_area

    return _ret
