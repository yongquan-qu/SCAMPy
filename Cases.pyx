import numpy as np
include "parameters.pxi"
import cython

from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
from TimeStepping cimport  TimeStepping
cimport Surface
cimport Forcing
cimport Radiation
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport *
import math as mt
from libc.math cimport sqrt, log, fabs,atan, exp, fmax
import netCDF4 as nc
from scipy.interpolate import interp1d

def CasesFactory(namelist, paramlist):
    if namelist['meta']['casename'] == 'Soares':
        return Soares(paramlist)
    elif namelist['meta']['casename'] == 'Nieuwstadt':
        return Nieuwstadt(paramlist)
    elif namelist['meta']['casename'] == 'Bomex':
        return Bomex(paramlist)
    elif namelist['meta']['casename'] == 'life_cycle_Tan2018':
        return life_cycle_Tan2018(paramlist)
    elif namelist['meta']['casename'] == 'Rico':
        return Rico(paramlist)
    elif namelist['meta']['casename'] == 'TRMM_LBA':
        return TRMM_LBA(paramlist)
    elif namelist['meta']['casename'] == 'ARM_SGP':
        return ARM_SGP(paramlist)
    elif namelist['meta']['casename'] == 'GATE_III':
        return GATE_III(paramlist)
    elif namelist['meta']['casename'] == 'DYCOMS_RF01':
        return DYCOMS_RF01(paramlist)
    elif namelist['meta']['casename'] == 'GABLS':
        return GABLS(paramlist)
    elif namelist['meta']['casename'] == 'SP':
        return SP(paramlist)
    elif namelist['meta']['casename'] == 'DryBubble':
        return DryBubble(paramlist)
    elif namelist['meta']['casename'] == 'LES_driven_SCM':
        return LES_driven_SCM(paramlist)

    else:
        print('case not recognized')
    return


cdef class CasesBase:
    def __init__(self, paramlist):
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_ts('Tsurface')
        Stats.add_ts('shf')
        Stats.add_ts('lhf')
        Stats.add_ts('ustar')
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_ts('Tsurface', self.Sur.Tsurface)
        Stats.write_ts('shf', self.Sur.shf)
        Stats.write_ts('lhf', self.Sur.lhf)
        Stats.write_ts('ustar', self.Sur.ustar)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        return


cdef class Soares(CasesBase):
#Soares, P.M.M., Miranda, P.M.A., Siebesma, A.P. and Teixeira, J. (2004),
#An eddy‐diffusivity/mass‐flux parametrization for dry and shallow cumulus convection.
#Q.J.R. Meteorol. Soc., 130: 3365-3383. doi:10.1256/qj.03.223
    def __init__(self, paramlist):
        self.casename = 'Soares2004'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingNone()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1000.0 * 100.0
        Ref.qtg = 5.0e-3
        Ref.Tg = 300.0
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] theta = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            double ql = 0.0, qi = 0.0
            Py_ssize_t k

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            if Gr.z_half[k] <= 1350.0:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4* Gr.z_half[k]/1000.0
                theta[k] = 300.0

            else:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4 * 1.35 - 9.4e-4 * (Gr.z_half[k]-1350.0)/1000.0
                theta[k] = 300.0 + 2.0 * (Gr.z_half[k]-1350.0)/1000.0
            GMV.U.values[k] = 0.01

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = theta[k]
                GMV.T.values[k] =  theta[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = theta[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = theta[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.zrough = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
        self.Sur.Tsurface = 300.0
        self.Sur.qsurface = 5.0e-3
        theta_flux = 6.0e-2
        qt_flux = 2.5e-5
        theta_surface = self.Sur.Tsurface
        self.Sur.lhf = qt_flux * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface) # It would be 0.0 if we follow Nieuwstadt.
        self.Sur.shf = theta_flux * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = False
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux   =  g * ((theta_flux + (eps_vi - 1.0) * (theta_surface * qt_flux + self.Sur.qsurface * theta_flux))
                                 / (theta_surface * (1.0 + (eps_vi-1) * self.Sur.qsurface)))
        self.Sur.initialize(Gr, TS)

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        return

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class Nieuwstadt(CasesBase):
# "Nieuwstadt, F. T., Mason, P. J., Moeng, C. H., & Schumann, U. (1993).
#Large-eddy simulation of the convective boundary layer: A comparison of
#four computer codes. In Turbulent shear flows 8 (pp. 343-367). Springer, Berlin, Heidelberg."
    def __init__(self, paramlist):
        self.casename = 'Nieuwstadt'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingNone()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1000.0 * 100.0
        Ref.qtg = 1.0e-12 #Total water mixing ratio at surface. if set to 0, alpha0, rho0, p0 are NaN (TBD)
        Ref.Tg = 300.0
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] theta = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            double ql = 0.0, qi = 0.0
            Py_ssize_t k

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            if Gr.z_half[k] <= 1350.0:
                GMV.QT.values[k] = 0.0
                theta[k] = 300.0

            else:
                GMV.QT.values[k] = 0.0
                theta[k] = 300.0 + 3.0 * (Gr.z_half[k]-1350.0)/1000.0
            GMV.U.values[k] = 0.01

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = theta[k]
                GMV.T.values[k] =  theta[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = theta[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = theta[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.zrough = 0.16 #1.0e-4 0.16 is the value specified in the Nieuwstadt paper.
        self.Sur.Tsurface = 300.0
        self.Sur.qsurface = 0.0
        theta_flux = 6.0e-2
        qt_flux = 0.0
        theta_surface = self.Sur.Tsurface
        self.Sur.lhf = 0.0 # It would be 0.0 if we follow Nieuwstadt.
        self.Sur.shf = theta_flux * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = False
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux   =  g * ((theta_flux + (eps_vi - 1.0) * (theta_surface * qt_flux + self.Sur.qsurface * theta_flux))
                                 / (theta_surface * (1.0 + (eps_vi-1) * self.Sur.qsurface)))
        self.Sur.initialize(Gr, TS)

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        return

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return

    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class Bomex(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Bomex'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 0.376e-4 # s^{-1}
        self.Fo.apply_subsidence = True
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.015e5  #Pressure at ground
        Ref.Tg = 300.4  #Temperature at ground
        Ref.qtg = 0.02245   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Bomex is cloud-free
            Py_ssize_t k

            theta_pert = 0.1
            qt_pert = 0.025/1000.0

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set Thetal profile
            if Gr.z_half[k] <= 520.:
                thetal[k] = 298.7
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                thetal[k] = 298.7 + (Gr.z_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000:
                thetal[k] = 302.4 + (Gr.z_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
            if Gr.z_half[k] > 2000.0:
                thetal[k] = 308.2 + (Gr.z_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

            #Set qt profile
            if Gr.z_half[k] <= 520:
                GMV.QT.values[k] = (17.0 + (Gr.z_half[k]) * (16.3-17.0)/520.0)/1000.0
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                GMV.QT.values[k] = (16.3 + (Gr.z_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000.0:
                GMV.QT.values[k] = (10.7 + (Gr.z_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
            if Gr.z_half[k] > 2000.0:
                GMV.QT.values[k] = (4.2 + (Gr.z_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0


            #Set u profile
            if Gr.z_half[k] <= 700.0:
                GMV.U.values[k] = -8.75
            if Gr.z_half[k] > 700.0:
                GMV.U.values[k] = -8.75 + (Gr.z_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

            #Set perturbations on qt and theta_l
            if Gr.z_half[k] <= 1600.0:
                thetal[k] = thetal[k] + theta_pert*(np.random.random_sample()-0.5)
                GMV.QT.values[k] = GMV.QT.values[k] + qt_pert*(np.random.random_sample()-0.5)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = 299.1 * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize(Gr, TS)
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = -10.0 + (1.8e-3)*Gr.z_half[k]
            # Set large-scale cooling
            if Gr.z_half[k] <= 1500.0:
                self.Fo.dTdt[k] =  (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])
            else:
                self.Fo.dTdt[k] = (-2.0/(3600 * 24.0) + (Gr.z_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])

            # Set large-scale drying
            if Gr.z_half[k] <= 300.0:
                self.Fo.dqtdt[k] = -1.2e-8   #kg/(kg * s)
            if Gr.z_half[k] > 300.0 and Gr.z_half[k] <= 500.0:
                self.Fo.dqtdt[k] = -1.2e-8 + (Gr.z_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

            #Set large scale subsidence
            if Gr.z_half[k] <= 1500.0:
                self.Fo.subsidence[k] = 0.0 + Gr.z_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
            if Gr.z_half[k] > 1500.0 and Gr.z_half[k] <= 2100.0:
                self.Fo.subsidence[k] = -0.65/100 + (Gr.z_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)
        return

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class life_cycle_Tan2018(CasesBase):
    # Taken from: "An extended eddy- diffusivity mass-flux scheme for unified representation of subgrid-scale turbulence and convection"
    # Tan, Z., Kaul, C. M., Pressel, K. G., Cohen, Y., Schneider, T., & Teixeira, J. (2018).
    #  Journal of Advances in Modeling Earth Systems, 10. https://doi.org/10.1002/2017MS001162

    def __init__(self, paramlist):
        self.casename = 'life_cycle_Tan2018'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 0.376e-4 # s^{-1}
        self.Fo.apply_subsidence = True
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.015e5  #Pressure at ground
        Ref.Tg = 300.4  #Temperature at ground
        Ref.qtg = 0.02245   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Bomex is cloud-free
        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set Thetal profile
            if Gr.z_half[k] <= 520.:
                thetal[k] = 298.7
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                thetal[k] = 298.7 + (Gr.z_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000:
                thetal[k] = 302.4 + (Gr.z_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
            if Gr.z_half[k] > 2000.0:
                thetal[k] = 308.2 + (Gr.z_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

            #Set qt profile
            if Gr.z_half[k] <= 520:
                GMV.QT.values[k] = (17.0 + (Gr.z_half[k]) * (16.3-17.0)/520.0)/1000.0
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                GMV.QT.values[k] = (16.3 + (Gr.z_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000.0:
                GMV.QT.values[k] = (10.7 + (Gr.z_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
            if Gr.z_half[k] > 2000.0:
                GMV.QT.values[k] = (4.2 + (Gr.z_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0


            #Set u profile
            if Gr.z_half[k] <= 700.0:
                GMV.U.values[k] = -8.75
            if Gr.z_half[k] > 700.0:
                GMV.U.values[k] = -8.75 + (Gr.z_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)


        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = 299.1 * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.lhf0 = self.Sur.lhf
        self.shf0 = self.Sur.shf
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux = (g * ((8.0e-3 + (eps_vi-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (eps_vi-1) * 22.45e-3))))
        self.Sur.initialize(Gr, TS)
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = -10.0 + (1.8e-3)*Gr.z_half[k]
            # Set large-scale cooling
            if Gr.z_half[k] <= 1500.0:
                self.Fo.dTdt[k] =  (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])
            else:
                self.Fo.dTdt[k] = (-2.0/(3600 * 24.0) + (Gr.z_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])
            # Set large-scale drying
            if Gr.z_half[k] <= 300.0:
                self.Fo.dqtdt[k] = -1.2e-8   #kg/(kg * s)
            if Gr.z_half[k] > 300.0 and Gr.z_half[k] <= 500.0:
                self.Fo.dqtdt[k] = -1.2e-8 + (Gr.z_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

            #Set large scale subsidence
            if Gr.z_half[k] <= 1500.0:
                self.Fo.subsidence[k] = 0.0 + Gr.z_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
            if Gr.z_half[k] > 1500.0 and Gr.z_half[k] <= 2100.0:
                self.Fo.subsidence[k] = -0.65/100 + (Gr.z_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)
        return

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        weight = 1.0
        weight_factor = 0.01 + 0.99 *(np.cos(2.0*pi * TS.t /3600.0) + 1.0)/2.0
        weight = weight * weight_factor
        self.Sur.lhf = self.lhf0*weight
        self.Sur.shf = self.shf0*weight
        self.Sur.bflux = (g * ((8.0e-3*weight + (eps_vi-1.0)*(299.1 * 5.2e-5*weight  + 22.45e-3 * 8.0e-3*weight)) /(299.1 * (1.0 + (eps_vi-1) * 22.45e-3))))
        self.Sur.update(GMV, TS)
        return
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class Rico(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Rico'
        self.Sur = Surface.SurfaceFixedCoeffs(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        cdef double latitude = 18.0
        self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = True
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.0154e5  #Pressure at ground
        Ref.Tg = 299.8  #Temperature at ground
        cdef double pvg = pv_star(Ref.Tg)
        Ref.qtg = eps_v * pvg/(Ref.Pg - pvg)   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Rico is cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] =  -9.9 + 2.0e-3 * Gr.z_half[k]
            GMV.V.values[k] = -3.8
            #Set Thetal profile
            if Gr.z_half[k] <= 740.0:
                thetal[k] = 297.9
            else:
                thetal[k] = 297.9 + (317.0-297.9)/(4000.0-740.0)*(Gr.z_half[k] - 740.0)

            #Set qt profile
            if Gr.z_half[k] <= 740.0:
                GMV.QT.values[k] =  (16.0 + (13.8 - 16.0)/740.0 * Gr.z_half[k])/1000.0
            elif Gr.z_half[k] > 740.0 and Gr.z_half[k] <= 3260.0:
                GMV.QT.values[k] = (13.8 + (2.4 - 13.8)/(3260.0-740.0) * (Gr.z_half[k] - 740.0))/1000.0
            else:
                GMV.QT.values[k] = (2.4 + (1.8-2.4)/(4000.0-3260.0)*(Gr.z_half[k] - 3260.0))/1000.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.00015
        self.Sur.cm  = 0.001229
        self.Sur.ch = 0.001094
        self.Sur.cq = 0.001133
        # Adjust for non-IC grid spacing
        grid_adjust = (np.log(20.0/self.Sur.zrough)/np.log(Gr.z_half[Gr.gw]/self.Sur.zrough))**2
        self.Sur.cm = self.Sur.cm * grid_adjust
        self.Sur.ch = self.Sur.ch * grid_adjust
        self.Sur.cq = self.Sur.cq * grid_adjust
        self.Sur.Tsurface = 299.8
        self.Sur.initialize(Gr, TS)
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        for k in xrange(Gr.nzg):
            # Geostrophic velocity profiles
            self.Fo.ug[k] = -9.9 + 2.0e-3 * Gr.z_half[k]
            self.Fo.vg[k] = -3.8
            # Set large-scale cooling
            self.Fo.dTdt[k] =  (-2.5/(3600.0 * 24.0))  * exner_c(Ref.p0_half[k])

            # Set large-scale moistening
            if Gr.z_half[k] <= 2980.0:
                self.Fo.dqtdt[k] =  (-1.0 + 1.3456/2980.0 * Gr.z_half[k])/86400.0/1000.0   #kg/(kg * s)
            else:
                self.Fo.dqtdt[k] = 0.3456/86400.0/1000.0

            #Set large scale subsidence
            if Gr.z_half[k] <= 2260.0:
                self.Fo.subsidence[k] = -(0.005/2260.0) * Gr.z_half[k]
            else:
                self.Fo.subsidence[k] = -0.005
        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class TRMM_LBA(CasesBase):
    # adopted from: "Daytime convective development over land- A model intercomparison based on LBA observations",
    # By Grabowski et al (2006)  Q. J. R. Meteorol. Soc. 132 317-344
    def __init__(self, paramlist):
        self.casename = 'TRMM_LBA'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.Rad = Radiation.RadiationTRMM_LBA()
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 991.3*100  #Pressure at ground
        Ref.Tg = 296.85   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        pvg = pv_star(Ref.Tg)
        Ref.qtg = eps_v * pvg/(Ref.Pg - pvg)#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:

            double [:] p1 = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # TRMM_LBA inputs from Grabowski et al. 2006
        z_in = np.array([0.130,  0.464,  0.573,  1.100,  1.653,  2.216,  2.760,
                         3.297,  3.824,  4.327,  4.787,  5.242,  5.686,  6.131,
                         6.578,  6.996,  7.431,  7.881,  8.300,  8.718,  9.149,
                         9.611, 10.084, 10.573, 11.008, 11.460, 11.966, 12.472,
                        12.971, 13.478, 13.971, 14.443, 14.956, 15.458, 16.019,
                        16.491, 16.961, 17.442, 17.934, 18.397, 18.851, 19.331,
                        19.809, 20.321, 20.813, 21.329, 30.000]) * 1000 - 130.0 #LES z is in meters

        p_in = np.array([991.3, 954.2, 942.0, 886.9, 831.5, 778.9, 729.8,
                         684.0, 641.7, 603.2, 570.1, 538.6, 509.1, 480.4,
                         454.0, 429.6, 405.7, 382.5, 361.1, 340.9, 321.2,
                         301.2, 281.8, 263.1, 246.1, 230.1, 213.2, 197.0,
                         182.3, 167.9, 154.9, 143.0, 131.1, 119.7, 108.9,
                         100.1,  92.1,  84.6,  77.5,  71.4,  65.9,  60.7,
                          55.9,  51.3,  47.2,  43.3,  10.3]) * 100 # LES pres is in pasc

        T_in = np.array([23.70,  23.30,  22.57,  19.90,  16.91,  14.09,  11.13,
                          8.29,   5.38,   2.29,  -0.66,  -3.02,  -5.28,  -7.42,
                        -10.34, -12.69, -15.70, -19.21, -21.81, -24.73, -27.76,
                        -30.93, -34.62, -38.58, -42.30, -46.07, -50.03, -54.67,
                        -59.16, -63.60, -67.68, -70.77, -74.41, -77.51, -80.64,
                        -80.69, -80.00, -81.38, -81.17, -78.32, -74.77, -74.52,
                        -72.62, -70.87, -69.19, -66.90, -66.90]) + 273.15 # LES T is in deg K

        RH_in = np.array([98.00,  86.00,  88.56,  87.44,  86.67,  83.67,  79.56,
                          84.78,  84.78,  89.33,  94.33,  92.00,  85.22,  77.33,
                          80.11,  66.11,  72.11,  72.67,  52.22,  54.67,  51.00,
                          43.78,  40.56,  43.11,  54.78,  46.11,  42.33,  43.22,
                          45.33,  39.78,  33.78,  28.78,  24.67,  20.67,  17.67,
                          17.11,  16.22,  14.22,  13.00,  13.00,  12.22,   9.56,
                           7.78,   5.89,   4.33,   3.00,   3.00])

        u_in = np.array([0.00,   0.81,   1.17,   3.44,   3.53,   3.88,   4.09,
                         3.97,   1.22,   0.16,  -1.22,  -1.72,  -2.77,  -2.65,
                        -0.64,  -0.07,  -1.90,  -2.70,  -2.99,  -3.66,  -5.05,
                        -6.64,  -4.74,  -5.30,  -6.07,  -4.26,  -7.52,  -8.88,
                        -9.00,  -7.77,  -5.37,  -3.88,  -1.15,  -2.36,  -9.20,
                        -8.01,  -5.68,  -8.83, -14.51, -15.55, -15.36, -17.67,
                       -17.82, -18.94, -15.92, -15.32, -15.32])

        v_in = np.array([-0.40,  -3.51,  -3.88,  -4.77,  -5.28,  -5.85,  -5.60,
                         -2.67,  -1.47,   0.57,   0.89,  -0.08,   1.11,   2.15,
                          3.12,   3.22,   3.34,   1.91,   1.15,   1.01,  -0.57,
                         -0.67,   0.31,   2.97,   2.32,   2.66,   4.79,   3.40,
                          3.14,   3.93,   7.57,   2.58,   2.50,   6.44,   6.84,
                          0.19,  -2.20,  -3.60,   0.56,   6.68,   9.41,   7.03,
                          5.32,   1.14,  -0.65,   5.27,   5.27])
        # interpolate to the model grid-points

        p1 = np.interp(Gr.z_half,z_in,p_in)
        GMV.U.values = np.interp(Gr.z_half,z_in,u_in)
        GMV.V.values = np.interp(Gr.z_half,z_in,v_in)

        # get the entropy from RH, p, T
        RH = np.zeros(Gr.nzg)
        RH[Gr.gw:Gr.nzg-Gr.gw] = np.interp(Gr.z_half[Gr.gw:Gr.nzg-Gr.gw],z_in,RH_in)
        RH[0] = RH[3]
        RH[1] = RH[2]
        RH[Gr.nzg-Gr.gw+1] = RH[Gr.nzg-Gr.gw-1]

        T = np.zeros(Gr.nzg)
        T[Gr.gw:Gr.nzg-Gr.gw] = np.interp(Gr.z_half[Gr.gw:Gr.nzg-Gr.gw],z_in,T_in)
        GMV.T.values = T
        theta_rho = RH*0.0
        epsi = 287.1/461.5
        cdef double PV_star # here pv_star is a function
        cdef double qv_star

        GMV.U.set_bcs(Gr)
        GMV.T.set_bcs(Gr)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            PV_star = pv_star(GMV.T.values[k])
            qv_star = PV_star*epsi/(p1[k]- PV_star + epsi*PV_star*RH[k]/100.0) # eq. 37 in pressel et al and the def of RH
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.QT.values[k] = qv_star*RH[k]/100.0
            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))
            theta_rho[k] = theta_rho_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)

        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        #self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = (273.15+23) * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize(Gr, TS)

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        return


    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.lhf = 554.0 * mt.pow(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.3)
        self.Sur.shf = 270.0 * mt.pow(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.5)
        self.Sur.update(GMV, TS)
        # fix momentum fluxes to zero as they are not used in the paper
        self.Sur.rho_uflux = 0.0
        self.Sur.rho_vflux = 0.0
        return

    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        self.Fo.update(GMV, TS)

        return

    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class ARM_SGP(CasesBase):
    # adopted from: "Large-eddy simulation of the diurnal cycle of shallow cumulus convection over land",
    # By Brown et al. (2002)  Q. J. R. Meteorol. Soc. 128, 1075-1093
    def __init__(self, paramlist):
        self.casename = 'ARM_SGP'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 8.5e-5
        self.Fo.apply_subsidence =False

        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 970.0*100 #Pressure at ground
        Ref.Tg = 299.0   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        Ref.qtg = 15.2/1000#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            Py_ssize_t k
            double [:] p1 = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # ARM_SGP inputs
        z_in = np.array([0.0, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0 ]) #LES z is in meters
        Theta_in = np.array([299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2]) # K
        r_in = np.array([15.2,15.17,14.98,14.8,14.7,13.5,3.0,3.0])/1000 # qt should be in kg/kg
        qt_in = np.divide(r_in,(1+r_in))

        # interpolate to the model grid-points
        Theta = np.interp(Gr.z_half,z_in,Theta_in)
        qt = np.interp(Gr.z_half,z_in,qt_in)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] = 10.0
            GMV.QT.values[k] = qt[k]
            GMV.T.values[k] = Theta[k]*exner_c(Ref.p0_half[k])
            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))


        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Tsurface = 299.0 * exner_c(Ref.Pg)
        self.Sur.qsurface = 15.2e-3 # kg/kg
        self.Sur.lhf = 5.0
        self.Sur.shf = -30.0
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize(Gr, TS)

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        cdef:
            Py_ssize_t k
        for k in xrange(Gr.nzg):
            self.Fo.ug[k] = 10.0
            self.Fo.vg[k] = 0.0

        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            double [:] t_Sur_in = np.array([0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5]) * 3600 #LES time is in sec
            double [:] SH = np.array([-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10]) # W/m^2
            double [:] LH = np.array([5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0]) # W/m^2
        self.Sur.shf = np.interp(TS.t,t_Sur_in,SH)
        self.Sur.lhf = np.interp(TS.t,t_Sur_in,LH)
        # if fluxes vanish bflux vanish and wstar and obukov length are NaNs
        ## CK +++ I commented out the lines below as I don't think this is how we want to fix things!
        # if self.Sur.shf < 1.0:
        #     self.Sur.shf = 1.0
        # if self.Sur.lhf < 1.0:
        #     self.Sur.lhf = 1.0
        #+++++++++
        self.Sur.update(GMV, TS)
        # fix momentum fluxes to zero as they are not used in the paper
        self.Sur.rho_uflux = 0.0
        self.Sur.rho_vflux = 0.0
        return

    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        cdef:
            double [:] t_in = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 14.5]) * 3600.0 #LES time is in sec
            double [:] AT_in = np.array([0.0, 0.0, 0.0, -0.08, -0.016, -0.016])/3600.0 # Advective forcing for theta [K/h] converted to [K/sec]
            double [:] RT_in = np.array([-0.125, 0.0, 0.0, 0.0, 0.0, -0.1])/3600.0  # Radiative forcing for theta [K/h] converted to [K/sec]
            double [:] Rqt_in = np.array([0.08, 0.02, 0.04, -0.1, -0.16, -0.3])/1000.0/3600.0 # Radiative forcing for qt converted to [kg/kg/sec]
            double dTdt = np.interp(TS.t,t_in,AT_in) + np.interp(TS.t,t_in,RT_in)
            double dqtdt =  np.interp(TS.t,t_in,Rqt_in)
        for k in xrange(self.Fo.Gr.nzg): # correct dims
                if self.Fo.Gr.z_half[k] <=1000.0:
                    self.Fo.dTdt[k] = dTdt
                    self.Fo.dqtdt[k]  = dqtdt * exner_c(self.Fo.Ref.p0_half[k])
                elif self.Fo.Gr.z_half[k] > 1000.0  and self.Fo.Gr.z_half[k] <= 2000.0:
                    self.Fo.dTdt[k] = dTdt*(1-(self.Fo.Gr.z_half[k]-1000.0)/1000.0)
                    self.Fo.dqtdt[k]  = dqtdt * exner_c(self.Fo.Ref.p0_half[k])\
                                        *(1-(self.Fo.Gr.z_half[k]-1000.0)/1000.0)
        self.Fo.update(GMV, TS)

        return

    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class GATE_III(CasesBase):
    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    def __init__(self, paramlist):
        self.casename = 'GATE_III'
        self.Sur = Surface.SurfaceFixedCoeffs(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_subsidence = False
        self.Fo.apply_coriolis = False

        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1013.0*100  #Pressure at ground
        Ref.Tg = 299.184   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        Ref.qtg = 16.5/1000#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double qv
            double [:] qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
            double [:] T = np.zeros((Gr.nzg,),dtype=np.double,order='c') # Gr.nzg = Gr.nz + 2*Gr.gw
            double [:] U = np.zeros((Gr.nzg,),dtype=np.double,order='c')
            double [:] theta_rho = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # GATE_III inputs - I extended them to z=22 km
        z_in  = np.array([ 0.0,   0.5,  1.0,  1.5,  2.0,   2.5,    3.0,   3.5,   4.0,   4.5,   5.0,  5.5,  6.0,  6.5,
                           7.0, 7.5, 8.0,  8.5,   9.0,   9.5, 10.0,   10.5,   11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
                           14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 27.0]) * 1000.0 #z is in meters
        r_in = np.array([16.5,  16.5, 13.5, 12.0, 10.0,   8.7,    7.1,   6.1,   5.2,   4.5,   3.6,  3.0,  2.3, 1.75, 1.3,
                         0.9, 0.5, 0.25, 0.125, 0.065, 0.003, 0.0015, 0.0007,  0.0003,  0.0001,  0.0001,  0.0001,  0.0001,
                         0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001])/1000 # mixing ratio should be in kg/kg
        U_in  = np.array([  -1, -1.75, -2.5, -3.6, -6.0, -8.75, -11.75, -13.0, -13.1, -12.1, -11.0, -8.5, -5.0, -2.6, 0.0,
                            0.5, 0.4,  0.3,   0.0,  -1.0, -2.5,   -3.5,   -4.5, -4.8, -5.0, -3.5, -2.0, -1.0, -1.0, -1.0,
                            -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0, -3.0])# [m/s]
        qt_in = np.divide(r_in,(1+r_in)) # convert mixing ratio to specific humidity

        # temperature is taken from a different input plot at different z levels
        T_in = np.array([299.184, 294.836, 294.261, 288.773, 276.698, 265.004, 253.930, 243.662, 227.674, 214.266, 207.757, 201.973, 198.278, 197.414, 198.110, 198.110])
        z_T_in = np.array([0.0, 0.492, 0.700, 1.698, 3.928, 6.039, 7.795, 9.137, 11.055, 12.645, 13.521, 14.486, 15.448, 16.436, 17.293, 22.0])*1000.0 # for km

        # interpolate to the model grid-points
        T = np.interp(Gr.z_half,z_T_in,T_in) # interpolate to ref pressure level
        qt = np.interp(Gr.z_half,z_in,qt_in)
        U = np.interp(Gr.z_half,z_in,U_in)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.QT.values[k] = qt[k]
            GMV.T.values[k] = T[k]
            GMV.U.values[k] = U[k]

            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))
        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.qsurface = 16.5/1000.0 # kg/kg
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.cm  = 0.0012
        self.Sur.ch = 0.0034337
        self.Sur.cq = 0.0034337
        self.Sur.Tsurface = 299.184
        self.Sur.initialize(Gr, TS)

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        #LES z is in meters
        z_in     = np.array([ 0.0,   0.5,  1.0,  1.5,   2.0,   2.5,    3.0,   3.5,   4.0,   4.5,   5.0,   5.5,   6.0,
                              6.5,  7.0,  7.5,   8.0,  8.5,   9.0,  9.5,  10.0,  10.5,  11.0,    11.5,   12.0, 12.5,
                              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0]) * 1000.0
        u_in     = np.array([  -1, -1.75, -2.5, -3.6,  -6.0, -8.75, -11.75, -12.9, -13.1, -12.1, -11.0,  -8.5,  -5.0,
                               -2.6,  0.0,  0.5,   0.4,  0.3,   0.0, -1.0,  -3.0,  -3.5,  -4.5,    -4.6,   -5.0, -3.5,
                               -2.0, -1.0, -1.0, -1.0, -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0])
        # Radiative forcing for T [K/d] converted to [K/sec]
        RAD_in   = np.array([-2.9,  -1.1, -0.8, -1.1, -1.25, -1.35,   -1.4,  -1.4, -1.44, -1.52,  -1.6, -1.54, -1.49,
                             -1.43, -1.36, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05,  -1.0,  -0.95,   -0.9,  -0.85, -0.8,
                             -0.75, -0.7, -0.6, -0.3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)
        # Advective qt forcing  for theta [g/kg/d] converted to [kg/kg/sec]
        r_tend_in = np.array([ 0.0,   1.2,  2.0,  2.3,   2.2,   2.1,    1.9,   1.7,   1.5,  1.35,  1.22,  1.08,  0.95,
                               0.82,  0.7,  0.6,   0.5,  0.4,   0.3,  0.2,   0.1,  0.05, 0.0025, 0.0012, 0.0006,  0.0,
                               0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)/1000.0
        # Radiative T forcing [K/d] converted to [K/sec]
        Ttend_in = np.array([ 0.0,  -1.0, -2.2, -3.0,  -3.5,  -3.8,   -4.0,  -4.1,  -4.2,  -4.2,  -4.1,  -4.0, -3.85,
                              -3.7, -3.5, -3.25, -3.0, -2.8,  -2.5, -2.1,  -1.7,  -1.3,   -1.0,   -0.7,   -0.5, -0.4,
                              -0.3, -0.2, -0.1,-0.05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)

        Qtend_in = np.divide(r_tend_in,(1+r_tend_in)) # convert mixing ratio to specific humidity

        self.Fo.dqtdt = np.interp(Gr.z_half,z_in,Qtend_in)
        self.Fo.dTdt = np.interp(Gr.z_half,z_in,Ttend_in) + np.interp(Gr.z_half,z_in,RAD_in)
        return


    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS) # here lhf and shf are needed for calcualtion of bflux in surface and thus u_star
        return

    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        self.Fo.update(GMV, TS)
        return

    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class DYCOMS_RF01(CasesBase):
    """
    see Stevens et al 2005:
    Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus.
    Mon. Wea. Rev., 133, 1443â€“1462.
    doi: http://dx.doi.org/10.1175/MWR2930.1
    """
    def __init__(self, paramlist):
        self.casename = 'DYCOMS_RF01'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingDYCOMS_RF01() # radiation is included in Forcing
        self.Rad = Radiation.RadiationDYCOMS_RF01()
        self.inversion_option = 'thetal_maxgrad'
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg   = 1017.8 * 100.0
        Ref.qtg  = 9.0 / 1000.0
        # Use an exner function with values for Rd, and cp given in Stevens 2005 to compute temperature
        Ref.Tg   = 289.0 * exner_c(Ref.Pg, kappa = dycoms_Rd / dycoms_cp)
        Ref.initialize(Gr, Stats, namelist)
        return

    # helper function
    def dycoms_compute_thetal(self, p_, T_, ql_):
        """
        Compute thetal using constants from Stevens et al 2005 DYCOMS case.
        :param p: pressure [Pa]
        :param T: temperature [K]
        :param ql: liquid water specific humidity
        :return: theta l
        """
        theta_ = T_ / exner_c(p_, kappa = dycoms_Rd / dycoms_cp)
        return theta_ * mt.exp(-1. * dycoms_L * ql_ / (dycoms_cp * T_))

    # helper function
    def dycoms_sat_adjst(self, p_, thetal_, qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        We can't use the default scampy function because of different values of cp, Rd and L
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''
        #Compute temperature
        t_1 = thetal_ * exner_c(p_, kappa = dycoms_Rd / dycoms_cp)
        #Compute saturation vapor pressure
        pv_star_1 = pv_star(t_1)
        #Compute saturation specific humidity
        qs_1 = qv_star_c(p_, qt_, pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - self.dycoms_compute_thetal(p_, t_1, ql_1)
            t_2 = t_1 + dycoms_L * ql_1 / dycoms_cp
            pv_star_2 = pv_star(t_2)
            qs_2 = qv_star_c(p_, qt_, pv_star_2)
            ql_2 = qt_ - qs_2

            while np.fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = pv_star(t_2)
                qs_2 = qv_star_c(p_, qt_, pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - self.dycoms_compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c') # helper variable to recalculate temperature
        ql     = np.zeros((Gr.nzg,), dtype=np.double, order='c') # DYCOMS case is saturated
        qi     = 0.0                                             # no ice

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # thetal profile as defined in DYCOMS
            if Gr.z_half[k] <= 840.0:
               thetal[k] = 289.0
            if Gr.z_half[k] > 840.0:
               thetal[k] = (297.5 + (Gr.z_half[k] - 840.0)**(1.0/3.0))

            # qt profile as defined in DYCOMS
            if Gr.z_half[k] <= 840.0:
               GMV.QT.values[k] = 9. / 1000.0
            if Gr.z_half[k] > 840.0:
               GMV.QT.values[k] = 1.5 / 1000.0

            # ql and T profile
            # (calculated by saturation adjustment using thetal and qt values provided in DYCOMS
            # and using Rd, cp and L constants as defined in DYCOMS)
            GMV.T.values[k], GMV.QL.values[k] = self.dycoms_sat_adjst(Ref.p0_half[k], thetal[k], GMV.QT.values[k])

            # thermodynamic variable profile (either entropy or thetal)
            # (calculated based on T and ql profiles.
            # Here we use Rd, cp and L constants as defined in scampy)
            GMV.THL.values[k] = t_to_thetali_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)
            if GMV.H.name == 'thetal':
                GMV.H.values[k] = t_to_thetali_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)
            elif GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)

            # buoyancy profile
            qv = GMV.QT.values[k] - qi - GMV.QL.values[k]
            rho = rho_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)
            GMV.B.values[k] = buoyancy_c(Ref.rho0_half[k], rho)

            # velocity profile (geostrophic)
            GMV.U.values[k] = 7.0
            GMV.V.values[k] = -5.5

        # fill out boundary conditions
        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.QL.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.THL.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.B.set_bcs(Gr)

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.zrough      = 1.0e-4
        self.Sur.ustar_fixed = False
        self.Sur.cm          = 0.0011

        # sensible heat flux
        self.Sur.shf = 15.0
        # latent heat flux
        self.Sur.lhf = 115.0

        self.Sur.Tsurface = 292.5    # K      # i.e. the SST from DYCOMS setup
        self.Sur.qsurface = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for scampy?
        #density_surface  = 1.22     # kg/m^3

        # buoyancy flux
        theta_flux       = self.Sur.shf / cpm_c(self.Sur.qsurface)        / Ref.rho0[Gr.gw-1]
        qt_flux          = self.Sur.lhf / latent_heat(self.Sur.Tsurface)  / Ref.rho0[Gr.gw-1]
        theta_surface    = self.Sur.Tsurface / exner_c(Ref.Pg)
        self.Sur.bflux   =  g * ((theta_flux + (eps_vi - 1.0) * (theta_surface * qt_flux + self.Sur.qsurface * theta_flux))
                                 / (theta_surface * (1.0 + (eps_vi-1) * self.Sur.qsurface)))
        self.Sur.Gr  = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize(Gr, TS)

        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)

        # geostrophic velocity profiles
        self.Fo.ug[:] = 7.0
        self.Fo.vg[:] = -5.5

        # large scale subsidence
        divergence = 3.75e-6    # divergence is defined twice: here and in __init__ of ForcingDYCOMS_RF01 class
                                # To be able to have self.Fo.divergence available here,
                                # we would have to change the signature of ForcingBase class
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            self.Fo.subsidence[k] = - Gr.z_half[k] * divergence

        # no large-scale drying
        self.Fo.dqtdt[:] = 0. #kg/(kg * s)

        # radiation is treated as a forcing term (see eq. 3 in Stevens et. al. 2005)
        # cloud-top cooling + cloud-base warming + cooling in free troposphere

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.Gr  = Gr
        self.Rad.initialize(Gr, GMV, TS)
        self.Rad.calculate_radiation(Ref, Gr, GMV, TS)
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        self.Rad.initialize_io(Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        self.Fo.io(Stats)
        self.Rad.io(Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class GABLS(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'GABLS'
        self.Sur = Surface.SurfaceMoninObukhovDry(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        cdef double latitude = 73.0
        self.Fo.coriolis_param = 1.39e-4 # s^{-1}
        # self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = False
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.0e5  #Pressure at ground
        Ref.Tg = 265.0  #Temperature at ground
        Ref.qtg = 1.0e-12 #Total water mixing ratio at surface. if set to 0, alpha0, rho0, p0 are NaN (TBD)
        Ref.initialize(Gr, Stats, namelist)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of GABLS cloud-free
            double [:] theta_pert = np.random.random_sample(Gr.nzg)
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set wind velocity profile
            GMV.U.values[k] =  8.0
            GMV.V.values[k] =  0.0

            #Set Thetal profile
            if Gr.z_half[k] <= 100.0:
                thetal[k] = 265.0
            else:
                thetal[k] = 265.0 + (Gr.z_half[k] - 100.0) * 0.01

            #Set qt profile
            GMV.QT.values[k] = 0.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k]) # No water content
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.1
        self.Sur.Tsurface = 265.0
        self.Sur.initialize(Gr, TS)
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            # Geostrophic velocity profiles.
            self.Fo.ug[k] = 8.0
            self.Fo.vg[k] = 0.0
        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.Tsurface = 265.0 - (0.25/3600.0)*TS.t
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

# Not fully implemented yet - Ignacio
cdef class SP(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'SP'
        self.Sur = Surface.SurfaceSullivanPatton(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 1.0e-4 # s^{-1}
        # self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = False
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.0e5  #Pressure at ground
        Ref.Tg = 300.0  #Temperature at ground
        Ref.qtg = 1.0e-4   #Total water mixing ratio at surface. if set to 0, alpha0, rho0, p0 are NaN.
        Ref.initialize(Gr, Stats, namelist)
        return

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of SP cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] =  1.0
            GMV.V.values[k] =  0.0
            #Set Thetal profile
            if Gr.z_half[k] <= 974.0:
                thetal[k] = 300.0
            elif Gr.z_half[k] < 1074.0:
                thetal[k] = 300.0 + (Gr.z_half[k] - 974.0) * 0.08
            else:
                thetal[k] = 308.0 + (Gr.z_half[k] - 1074.0) * 0.003

            #Set qt profile
            GMV.QT.values[k] = 0.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.1
        self.Sur.Tsurface = 300.0
        theta_surface    = self.Sur.Tsurface / exner_c(Ref.Pg)
        theta_flux = 0.24
        self.Sur.bflux   =  g * theta_flux / theta_surface
        # self.Sur.bflux = 0.24 * exner_c(Ref.p0_half[Gr.gw]) * g / (Ref.p0_half[Gr.gw]*Ref.alpha0_half[Gr.gw]/Rd)
        self.Sur.initialize(Gr, TS)
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = 1.0
            self.Fo.vg[k] = 0.0
        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class DryBubble(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'DryBubble'
        self.Sur = Surface.SurfaceNone()
        self.Fo = Forcing.ForcingNone()
        self.Rad = Radiation.RadiationNone()
        self.inversion_option = 'theta_rho'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.Pg = 1.0e5  #Pressure at ground
        Ref.qtg = 1.0e-5
        Ref.Tg = 296
        Ref.initialize(Gr, Stats, namelist)
        return

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            Py_ssize_t k

        for k in xrange(Gr.nzg):
            GMV.U.values[k] = 0.01

        n_updrafts = 1
        # initialize Grid Mean Profiles of thetali and qt
        z_in = np.array([
                          25.,   75.,  125.,  175.,  225.,  275.,  325.,  375.,  425.,
                         475.,  525.,  575.,  625.,  675.,  725.,  775.,  825.,  875.,
                         925.,  975., 1025., 1075., 1125., 1175., 1225., 1275., 1325.,
                        1375., 1425., 1475., 1525., 1575., 1625., 1675., 1725., 1775.,
                        1825., 1875., 1925., 1975., 2025., 2075., 2125., 2175., 2225.,
                        2275., 2325., 2375., 2425., 2475., 2525., 2575., 2625., 2675.,
                        2725., 2775., 2825., 2875., 2925., 2975., 3025., 3075., 3125.,
                        3175., 3225., 3275., 3325., 3375., 3425., 3475., 3525., 3575.,
                        3625., 3675., 3725., 3775., 3825., 3875., 3925., 3975., 4025.,
                        4075., 4125., 4175., 4225., 4275., 4325., 4375., 4425., 4475.,
                        4525., 4575., 4625., 4675., 4725., 4775., 4825., 4875., 4925.,
                        4975., 5025., 5075., 5125., 5175., 5225., 5275., 5325., 5375.,
                        5425., 5475., 5525., 5575., 5625., 5675., 5725., 5775., 5825.,
                        5875., 5925., 5975., 6025., 6075., 6125., 6175., 6225., 6275.,
                        6325., 6375., 6425., 6475., 6525., 6575., 6625., 6675., 6725.,
                        6775., 6825., 6875., 6925., 6975., 7025., 7075., 7125., 7175.,
                        7225., 7275., 7325., 7375., 7425., 7475., 7525., 7575., 7625.,
                        7675., 7725., 7775., 7825., 7875., 7925., 7975., 8025., 8075.,
                        8125., 8175., 8225., 8275., 8325., 8375., 8425., 8475., 8525.,
                        8575., 8625., 8675., 8725., 8775., 8825., 8875., 8925., 8975.,
                        9025., 9075., 9125., 9175., 9225., 9275., 9325., 9375., 9425.,
                        9475., 9525., 9575., 9625., 9675., 9725., 9775., 9825., 9875.,
                        9925., 9975.
        ])
        thetali_in = np.array([
                        299.9834, 299.9836, 299.9841, 299.985 , 299.9864, 299.9883,
                        299.9907, 299.9936, 299.9972, 300.0012, 300.0058, 300.011 ,
                        300.0166, 300.0228, 300.0293, 300.0363, 300.0436, 300.0512,
                        300.0591, 300.0672, 300.0755, 300.0838, 300.0921, 300.1004,
                        300.1086, 300.1167, 300.1245, 300.132 , 300.1393, 300.1461,
                        300.1525, 300.1583, 300.1637, 300.1685, 300.1726, 300.1762,
                        300.179 , 300.1812, 300.1826, 300.1833, 300.1833, 300.1826,
                        300.1812, 300.179 , 300.1762, 300.1727, 300.1685, 300.1637,
                        300.1584, 300.1525, 300.1461, 300.1393, 300.1321, 300.1245,
                        300.1167, 300.1087, 300.1005, 300.0922, 300.0838, 300.0755,
                        300.0673, 300.0592, 300.0513, 300.0437, 300.0364, 300.0294,
                        300.0228, 300.0167, 300.0111, 300.0059, 300.0013, 299.9972,
                        299.9937, 299.9908, 299.9884, 299.9865, 299.9851, 299.9842,
                        299.9837, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9835, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9835, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9835, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9835, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9835, 299.9835, 299.9835, 299.9835, 299.9835, 299.9835,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9836, 299.9836,
                        299.9836, 299.9836, 299.9836, 299.9836, 299.9837, 299.9837,
                        299.9837, 299.9837, 299.9837, 299.9837, 299.9837, 299.9837,
                        299.9837, 299.9837, 299.9837, 299.9837, 299.9837, 299.9837,
                        299.9837, 299.9837, 299.9837, 299.9837, 299.9837, 299.9837,
                        299.9837, 299.9837
        ])
                           #LES temperature_mean in K
        thetali = np.zeros(Gr.nzg)
        thetali[Gr.gw:Gr.nzg-Gr.gw] = np.interp(Gr.z_half[Gr.gw:Gr.nzg-Gr.gw],z_in,thetali_in)
        GMV.THL.values = thetali
        GMV.H.values = thetali
        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.QT.values[k] = 0.0

        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.qsurface = 1.0e-5
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.initialize(Gr, TS)
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        return
    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        self.Rad.initialize_io(Stats)
        self.Fo.initialize_io(Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        self.Rad.io(Stats)
        self.Fo.io(Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return
    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return

cdef class LES_driven_SCM(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'LES_driven_SCM'
        self.Sur = Surface.SurfaceLES(paramlist)
        self.Fo = Forcing.ForcingLES()
        self.Rad = Radiation.RadiationLES()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = False
        # get LES latitiude
        self.Fo.apply_subsidence = True
        self.Fo.nudge_tau = paramlist['forcing']['nudging_timescale']
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats, namelist):
        Ref.initialize(Gr, Stats, namelist)
        return

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            Py_ssize_t k

        les_data = nc.Dataset(Gr.les_filename,'r')
        z_les_half = np.array(les_data.groups['reference'].variables['zp_half'])
        thetali = np.array(les_data.groups['profiles'].variables['thetali_mean'])
        qt      = np.array(les_data.groups['profiles'].variables['qt_mean'])
        u_mean  = np.array(les_data.groups['profiles'].variables['u_mean'])
        v_mean  = np.array(les_data.groups['profiles'].variables['v_mean'])
        # interp1d from LES to SCM
        f_thetali     = interp1d(z_les_half, thetali[0,:], fill_value="extrapolate")
        GMV.H.values  = f_thetali(Gr.z_half)
        f_qt          = interp1d(z_les_half, qt[0,:], fill_value="extrapolate")
        GMV.QT.values = f_qt(Gr.z_half)
        f_u_mean      = interp1d(z_les_half, u_mean[0,:], fill_value="extrapolate")
        GMV.U.values  = f_u_mean(Gr.z_half)
        f_v_mean      = interp1d(z_les_half, v_mean[0,:], fill_value="extrapolate")
        GMV.V.values  = f_v_mean(Gr.z_half)

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref,  TimeStepping TS, namelist):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.qsurface = 1.0e-5
        self.Sur.zrough = 1.0e-4
        self.Sur.initialize(Gr, TS)
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(Gr, GMV, TS)
        return

    cpdef initialize_radiation(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, TimeStepping TS):
        self.Rad.Gr = Gr
        self.Rad.initialize(Gr, GMV, TS)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        self.Rad.initialize_io(Stats)
        self.Fo.initialize_io(Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        self.Rad.io(Stats)
        self.Fo.io(Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV, TS)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV, TS)
        return

    cpdef update_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV,  TimeStepping TS):
        self.Rad.update(Ref, Gr, GMV, TS)
        return
