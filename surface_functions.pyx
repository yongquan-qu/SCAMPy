import numpy as np
from thermodynamic_functions cimport latent_heat, pd_c, pv_c, sd_c, sv_c, cpm_c, theta_rho_c
from libc.math cimport acos, sqrt, cbrt, fabs, cos
include "parameters.pxi"

#Adapted from PyCLES: https://github.com/pressel/pycles

cdef double buoyancy_flux(double shf, double lhf, double T_b, double qt_b, double alpha0_0):
    cdef:
        double cp_ = cpm_c(qt_b)
        double lv = latent_heat(T_b)
    return (g * alpha0_0 / cp_ / T_b * (shf + (eps_vi-1.0) * cp_ * T_b * lhf /lv))

cdef inline double psi_m_unstable(double zeta, double zeta0):
    cdef double x = (1.0 - gamma_m * zeta)**0.25
    cdef double x0 = (1.0 - gamma_m * zeta0)**0.25
    cdef double psi_m = (2.0 * np.log((1.0 + x)/(1.0 + x0)) + np.log((1.0 + x*x)/(1.0 + x0 * x0))
                         -2.0 * np.arctan(x) + 2.0 * np.arctan(x0))
    return psi_m

cdef  inline double psi_h_unstable(double zeta, double zeta0):
    cdef double y = np.sqrt(1.0 - gamma_h * zeta )
    cdef double y0 = np.sqrt(1.0 - gamma_h * zeta0 )
    cdef double psi_h = 2.0 * np.log((1.0 + y)/(1.0 + y0))
    return psi_h


cdef inline double psi_m_stable(double zeta, double zeta0):
    cdef double psi_m = -beta_m * (zeta - zeta0)
    return  psi_m

cdef inline double psi_h_stable(double zeta, double zeta0):
    cdef double psi_h = -beta_h * (zeta - zeta0)
    return  psi_h


cpdef double entropy_flux(tflux,qtflux, p0_1, T_1, qt_1):
        cdef:
            double cp_1 = cpm_c(qt_1)
            double pd_1 = pd_c(p0_1, qt_1, qt_1)
            double pv_1 = pv_c(p0_1, qt_1, qt_1)
            double sd_1 = sd_c(pd_1, T_1)
            double sv_1 = sv_c(pv_1, T_1)
        return cp_1*tflux/T_1 + qtflux*(sv_1-sd_1)




cpdef double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) :

    cdef:
        double lmo, zeta, zeta0, psi_m,ustar
        double ustar0, ustar1, ustar_new, f0, f1, delta_ustar
        double logz = np.log(z1 / z0)
    #use neutral condition as first guess
    ustar0 = windspeed * vkb / logz
    ustar = ustar0
    if (np.abs(buoyancy_flux) > 1.0e-20):
        lmo = -ustar0 * ustar0 * ustar0 / (buoyancy_flux * vkb)
        zeta = z1 / lmo
        zeta0 = z0 / lmo
        if (zeta >= 0.0):
            f0 = windspeed - ustar0 / vkb * (logz - psi_m_stable(zeta, zeta0))
            ustar1 = windspeed * vkb / (logz - psi_m_stable(zeta, zeta0))
            lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
            zeta = z1 / lmo
            zeta0 = z0 / lmo
            f1 = windspeed - ustar1 / vkb * (logz - psi_m_stable(zeta, zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while np.abs(delta_ustar) > 1e-3:
                ustar_new = ustar1 - f1 * delta_ustar / (f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
                zeta = z1 / lmo
                zeta0 = z0 / lmo
                f1 = windspeed - ustar1 / vkb * (logz - psi_m_stable(zeta, zeta0))
                delta_ustar = ustar1 -ustar0

            ustar = ustar1
            
        else: # b_flux nonzero, zeta  is negative
            f0 = windspeed - ustar0 / vkb * (logz - psi_m_unstable(zeta, zeta0))
            ustar1 = windspeed * vkb / (logz - psi_m_unstable(zeta, zeta0))
            lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
            zeta = z1 / lmo
            zeta0 = z0 / lmo
            f1 = windspeed - ustar1 / vkb * (logz - psi_m_unstable(zeta, zeta0))
            ustar = ustar1
            delta_ustar = ustar1 - ustar0
            while np.abs(delta_ustar) > 1e-3:
                ustar_new = ustar1 - f1 * delta_ustar / (f1 - f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
                zeta = z1 / lmo
                zeta0 = z0 / lmo
                f1 = windspeed - ustar1 / vkb * (logz - psi_m_unstable(zeta, zeta0))
                delta_ustar = ustar1 - ustar0

            ustar = ustar1

    return ustar

cdef void exchange_coefficients_byun(double Ri, double zb, double z0, double *cm, double *ch, double *lmo):

    #Monin-Obukhov similarity based on
    #Daewon W. Byun, 1990: On the Analytical Solutions of Flux-Profile Relationships for the Atmospheric Surface Layer. J. Appl. Meteor., 29, 652–657.
    #doi: http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2
    cdef:
        double logz = np.log(zb/z0)
        double zfactor = zb/(zb-z0)*logz
        double zeta, zeta0, psi_m, psi_h
        double sb = Ri/Pr0
        double qb, pb, crit, angle, tb
        double  cu, cth


    if Ri > 0.0:
        zeta = zfactor/(2.0*beta_h*(beta_m*Ri -1.0))*((1.0-2.0*beta_h*Ri)-sqrt(1.0+4.0*(beta_h - beta_m)*sb))
        lmo[0] = zb/zeta
        zeta0 = z0/lmo[0]
        psi_m = psi_m_stable(zeta, zeta0)
        psi_h = psi_h_stable(zeta,zeta0)
    else:
        qb = 1.0/9.0 * (1.0 /(gamma_m * gamma_m) + 3.0 * gamma_h/gamma_m * sb * sb)
        pb = 1.0/54.0 * (-2.0/(gamma_m*gamma_m*gamma_m) + 9.0/gamma_m * (-gamma_h/gamma_m + 3.0)*sb * sb)
        crit = qb * qb *qb - pb * pb
        if crit < 0.0:
            tb = cbrt(sqrt(-crit) + fabs(pb))
            zeta = zfactor * (1.0/(3.0*gamma_m)-(tb + qb/tb))
        else:
            angle = acos(pb/sqrt(qb * qb * qb))
            zeta = zfactor * (-2.0 * sqrt(qb) * cos(angle/3.0)+1.0/(3.0*gamma_m))
        lmo[0] = zb/zeta
        zeta0 = z0/lmo[0]
        psi_m = psi_m_unstable(zeta, zeta0)
        psi_h = psi_h_unstable(zeta,zeta0)

    cu = vkb/(logz-psi_m)
    cth = vkb/(logz-psi_h)/Pr0
    cm[0] = cu * cu
    ch[0] = cu * cth
    return
