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
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        self.dTdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dqtdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return


cdef class RadiationNone(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        RadiationBase.initialize(self, Gr, GMV, TS)
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return



cdef class RadiationTRMM_LBA(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        RadiationBase.initialize(self, Gr, GMV, TS)
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double qv

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
            if self.apply_subsidence:
                # Apply large-scale subsidence tendencies
                GMV.H.subsidence[k] =  -(GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.subsidence[k]
                GMV.QT.subsidence[k] = -(GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.subsidence[k]
            else:
                GMV.H.subsidence[k] =  0.0
                GMV.QT.subsidence[k] = 0.0

            GMV.H.tendencies[k]  += GMV.H.subsidence[k]
            GMV.QT.tendencies[k] += GMV.QT.subsidence[k]


        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return


cdef class RadiationDYCOMS_RF01(RadiationBase):

    def __init__(self):
        RadiationBase.__init__(self)
        return

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        RadiationBase.initialize(self, Gr, GMV, TS)

        self.alpha_z    = 1.
        self.kappa      = 85.
        self.F0         = 70.
        self.F1         = 22.
        self.divergence = 3.75e-6  # divergence is defined twice: here and in initialize_forcing method of DYCOMS_RF01 case class
                                   # where it is used to initialize large scale subsidence

        self.f_rad = np.zeros((self.Gr.nzg + 1), dtype=np.double, order='c') # radiative flux at cell edges
        return

        """
        see eq. 3 in Stevens et. al. 2005 DYCOMS paper
        """
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double qv
            double zi
            double rhoi

        # find zi (level of 8.0 g/kg isoline of qt)
        for k in xrange(self.Gr.gw, self.Gr.nzg - self.Gr.gw):
            if (GMV.QT.values[k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                zi     = self.Gr.z[idx_zi]
                rhoi   = self.Ref.rho0[idx_zi]
                break

        # cloud-top cooling
        q_0 = 0.0
    
        self.f_rad = np.zeros((self.Gr.nzg + 1), dtype=np.double, order='c')
        self.f_rad[self.Gr.nzg] = self.F0 * np.exp(-q_0)
        for k in xrange(self.Gr.nzg - 1, -1, -1):
            q_0           += self.kappa * self.Ref.rho0_half[k] * GMV.QL.values[k] * self.Gr.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[0] += self.F1 * np.exp(-q_1)
        for k in xrange(1, self.Gr.nzg + 1):
            q_1           += self.kappa * self.Ref.rho0_half[k - 1] * GMV.QL.values[k - 1] * self.Gr.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in xrange(0, self.Gr.nzg):
            if self.Gr.z[k] > zi:
                cbrt_z         = cbrt(self.Gr.z[k] - zi)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)
        # condition at the top
        cbrt_z                   = cbrt(self.Gr.z[k] + self.Gr.dz - zi)
        self.f_rad[self.Gr.nzg] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)

        for k in xrange(self.Gr.gw, self.Gr.nzg - self.Gr.gw):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / self.Gr.dz / self.Ref.rho0_half[k] / dycoms_cp


        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply radiative temperature tendency
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k]  += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('rad_dTdt', self.dTdt[ self.Gr.gw     : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('rad_flux', self.f_rad[self.Gr.gw + 1 : self.Gr.nzg - self.Gr.gw + 1])
        return


cdef class RadiationLES(RadiationBase):
    def __init__(self, paramlist):
        RadiationBase.__init__(self)
        self.nudge_tau = paramlist['radiation']['nudging_timescale']
        return

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        RadiationBase.initialize(self, Gr, GMV, TS)
        les_data = nc.Dataset(Gr.les_filename,'r')
        t_les       = np.array(les_data.groups['profiles'].variables['t'])
        z_les       = np.array(les_data.groups['profiles'].variables['z'])
        les_dtdt_rad    = np.array(les_data['profiles'].variables['dtdt_rad'])
        t_scm = np.linspace(0.0,TS.t_max, int(TS.t_max/TS.dt)+1)

        f_dtdt_rad = interp2d(z_les, t_les, les_dtdt_rad)
        self.dtdt_rad = f_dtdt_rad(Gr.z_half, t_scm)
        return

    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t i, k

        i = int(TS.t/TS.dt)
        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.radiation[k] = self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], qv, self.dtdt_rad[i,k])
            GMV.H.tendencies[k] += GMV.H.radiation[k]
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        return