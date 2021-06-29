#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from Variables cimport GridMeanVariables, VariablePrognostic
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
from TimeStepping cimport TimeStepping
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin
import netCDF4 as nc
from scipy.interpolate import interp2d

cdef class ForcingBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        self.subsidence = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dTdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dqtdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.ug = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.vg = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
        for k in xrange(gw, self.Gr.nzg-gw):
            U.tendencies[k] -= self.coriolis_param * (self.vg[k] - V.values[k])
            V.tendencies[k] += self.coriolis_param * (self.ug[k] - U.values[k])
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return


cdef class ForcingNone(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        ForcingBase.initialize(self, Gr, GMV, TS)
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return



cdef class ForcingStandard(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        ForcingBase.initialize(self, Gr, GMV, TS)
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


        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return

cdef class ForcingDYCOMS_RF01(ForcingBase):

    def __init__(self):
        ForcingBase.__init__(self)
        return

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        ForcingBase.initialize(self, Gr, GMV, TS)

        self.alpha_z    = 1.
        self.kappa      = 85.
        self.F0         = 70.
        self.F1         = 22.
        self.divergence = 3.75e-6  # divergence is defined twice: here and in initialize_forcing method of DYCOMS_RF01 case class
                                   # where it is used to initialize large scale subsidence

        self.f_rad = np.zeros((self.Gr.nzg + 1), dtype=np.double, order='c') # radiative flux at cell edges
        return

    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        ForcingBase.coriolis_force(self, U, V)
        return

    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double qv

        # self.calculate_radiation(GMV)

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            GMV.QT.tendencies[k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            GMV.H.subsidence[k]  = -(GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.subsidence[k]
            GMV.QT.subsidence[k] = -(GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.subsidence[k]
            GMV.H.tendencies[k]  += GMV.H.subsidence[k]
            GMV.QT.tendencies[k] += GMV.QT.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('rad_dTdt', self.dTdt[ self.Gr.gw     : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('rad_flux', self.f_rad[self.Gr.gw + 1 : self.Gr.nzg - self.Gr.gw + 1])
        return


cdef class ForcingLES(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return

    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        ForcingBase.initialize(self, Gr, GMV, TS)
        # load the netCDF file
        les_data = nc.Dataset(Gr.les_filename,'r')
        t_les       = np.array(les_data.groups['profiles'].variables['t'])
        z_les       = np.array(les_data.groups['profiles'].variables['z'])
        les_dtdt_rad    = np.array(les_data['profiles'].variables['dtdt_rad'])
        les_dtdt_hadv   = np.array(les_data['profiles'].variables['dtdt_hadv'])
        les_dtdt_nudge  = np.array(les_data['profiles'].variables['dtdt_nudge'])
        les_dqtdt_hadv  = np.array(les_data['profiles'].variables['dqtdt_hadv'])
        les_dqtdt_nudge = np.array(les_data['profiles'].variables['dqtdt_nudge'])
        les_subsidence  = np.array(les_data['profiles'].variables['ls_subsidence'])
        les_u_nudge     = np.array(les_data['profiles'].variables['u_mean'])
        les_v_nudge     = np.array(les_data['profiles'].variables['v_mean'])

        t_scm = np.linspace(0.0,TS.t_max, int(TS.t_max/TS.dt)+1)

        # interp2d from LES to SCM
        f_dtdt_hadv = interp2d(z_les, t_les, les_dtdt_hadv)
        f_dtdt_nudge = interp2d(z_les, t_les, les_dtdt_nudge)
        f_dqtdt_hadv = interp2d(z_les, t_les, les_dqtdt_hadv)
        f_dqtdt_nudge = interp2d(z_les, t_les, les_dqtdt_nudge)

        self.dtdt_hadv = f_dtdt_hadv(Gr.z_half, t_scm)
        self.dtdt_nudge = f_dtdt_nudge(Gr.z_half, t_scm)
        self.dqtdt_hadv = f_dqtdt_hadv(Gr.z_half, t_scm)
        self.dqtdt_nudge = f_dqtdt_nudge(Gr.z_half, t_scm)

        f_u_nudge = interp2d(z_les, t_les, les_u_nudge)
        f_v_nudge = interp2d(z_les, t_les, les_v_nudge)
        self.u_nudge = f_u_nudge(Gr.z_half, t_scm)
        self.v_nudge = f_v_nudge(Gr.z_half, t_scm)

        f_subsidence = interp2d(z_les, t_les, les_subsidence)
        self.scm_subsidence = f_subsidence(Gr.z_half, t_scm)

        # get site's latitude
        sitedata = nc.Dataset('LES_driven_SCM/geolocation.nc','r')
        lats = np.array(sitedata.variables['lat'])
        latitude = lats[int(Gr.les_filename[29:31])-1]
        self.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}

        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        # read radiaitve forcing variables
        cdef:
            Py_ssize_t i, k

        i = int(TS.t/TS.dt)
        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.horz_adv[k] = self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], qv,self.dtdt_hadv[i,k])
            GMV.H.nudge[k] = self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], qv,self.dtdt_nudge[i,k])
            GMV.QT.horz_adv[k] = self.dqtdt_hadv[i,k]
            GMV.QT.nudge[k] = self.dqtdt_nudge[i,k]
            GMV.U.nudge[k] = (self.u_nudge[0,k] - GMV.U.values[k])/self.nudge_tau
            GMV.V.nudge[k] = (self.v_nudge[0,k] - GMV.V.values[k])/self.nudge_tau
            if self.apply_subsidence:
                # Apply large-scale subsidence tendencies
                GMV.H.subsidence[k] =  -(GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.scm_subsidence[i,k]
                GMV.QT.subsidence[k] = -(GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.scm_subsidence[i,k]
            else:
                GMV.H.subsidence[k] =  0.0
                GMV.QT.subsidence[k] = 0.0

            GMV.H.tendencies[k] += GMV.H.horz_adv[k] + GMV.H.nudge[k] + GMV.H.subsidence[k]
            GMV.QT.tendencies[k] += GMV.QT.horz_adv[k] + GMV.QT.nudge[k] + GMV.QT.subsidence[k]
            GMV.U.tendencies[k] += GMV.U.nudge[k]
            GMV.V.tendencies[k] += GMV.V.nudge[k]
        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return
