#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

import sys
import numpy as np
import cython
from Grid cimport Grid
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from ReferenceState cimport ReferenceState
from libc.math cimport fmax, fmin

from thermodynamic_functions cimport *

cdef class VariablePrognostic:
    def __init__(self,nz_tot,loc, kind, bc, name, units):
        # Value at the current timestep
        self.values = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Value at the next timestep, used for calculating turbulence tendencies
        self.new = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.mf_update = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.tendencies = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.radiation = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.horz_adv = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.nudge = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.fluc = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.subsidence = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Placement on staggered grid
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return

    cpdef zero_tendencies(self, Grid Gr):
        cdef:
            Py_ssize_t k
        with nogil:
            for k in xrange(Gr.nzg):
                self.tendencies[k] = 0.0
        return

    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        if self.bc == 'sym':
            for k in xrange(Gr.gw):
                self.values[start_high + k +1] = self.values[start_high  - k]
                self.values[start_low - k] = self.values[start_low + 1 + k]

                self.mf_update[start_high + k +1] = self.mf_update[start_high  - k]
                self.mf_update[start_low - k] = self.mf_update[start_low + 1 + k]

                self.new[start_high + k +1] = self.new[start_high  - k]
                self.new[start_low - k] = self.new[start_low + 1 + k]
        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0

            self.mf_update[start_high] = 0.0
            self.mf_update[start_low] = 0.0

            self.new[start_high] = 0.0
            self.new[start_low] = 0.0

            for k in xrange(1,Gr.gw):
                self.values[start_high+ k] = -self.values[start_high - k ]
                self.values[start_low- k] = -self.values[start_low + k  ]

                self.mf_update[start_high+ k] = -self.mf_update[start_high - k ]
                self.mf_update[start_low- k] = -self.mf_update[start_low + k  ]

                self.new[start_high+ k] = -self.new[start_high - k ]
                self.new[start_low- k] = -self.new[start_low + k  ]

        return

cdef class VariableDiagnostic:

    def __init__(self,nz_tot,loc, kind, bc, name, units):
        # Value at the current timestep
        self.values = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Placement on staggered grid
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return

    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw

        if self.bc == 'sym':
            for k in xrange(Gr.gw):
                self.values[start_high + k] = self.values[start_high  - 1]
                self.values[start_low - k] = self.values[start_low + 1]

        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0
            for k in xrange(1,Gr.gw):
                self.values[start_high+ k] = 0.0  #-self.values[start_high - k ]
                self.values[start_low- k] = 0.0 #-self.values[start_low + k ]

        return

cdef class GridMeanVariables:
    def __init__(self, namelist, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref

        self.lwp = 0.
        self.cloud_base   = 0.
        self.cloud_top    = 0.
        self.cloud_cover  = 0.

        self.U = VariablePrognostic(Gr.nzg, 'half', 'velocity', 'sym','u', 'm/s' )
        self.V = VariablePrognostic(Gr.nzg, 'half', 'velocity','sym', 'v', 'm/s' )
        # Just leave this zero for now!
        self.W = VariablePrognostic(Gr.nzg, 'full', 'velocity','asym', 'v', 'm/s' )

        # Create thermodynamic variables
        self.QT = VariablePrognostic(Gr.nzg, 'half', 'scalar','sym', 'qt', 'kg/kg')
        self.RH = VariablePrognostic(Gr.nzg, 'half', 'scalar','sym', 'RH', '%')

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = VariablePrognostic(Gr.nzg, 'half', 'scalar', 'sym','s', 'J/kg/K' )
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = VariablePrognostic(Gr.nzg, 'half', 'scalar', 'sym','thetal', 'K')
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal
        else:
            sys.exit('Did not recognize thermal variable ' + namelist['thermodynamics']['thermal_variable'])

        # Diagnostic Variables--same class as the prognostic variables, but we append to diagnostics list
        # self.diagnostics_list  = []
        self.QL  = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'ql',              'kg/kg')
        self.T   = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'temperature',     'K')
        self.B   = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'buoyancy',        'm^2/s^3')
        self.THL = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'thetal',          'K')

        self.cloud_fraction  = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'cloud fraction', '-')

        # TKE   TODO   repeated from EDMF_Environment.pyx logic
        if  namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            self.calc_tke = True
        else:
            self.calc_tke = False
        try:
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            pass

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['sgs'])
        except:
            self.EnvThermo_scheme = 'mean'

        #Now add the 2nd moment variables
        if self.calc_tke:
            self.TKE = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'tke','m^2/s^2' )
            self.W_third_m = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'W_third_m', 'm^3/s^3')

        if self.calc_scalar_var:
            self.QTvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'qt_var','kg^2/kg^2' )
            self.QT_third_m = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'qt_third_m','kg^3/kg^3' )
            if namelist['thermodynamics']['thermal_variable'] == 'entropy':
                self.Hvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 's_var', '(J/kg/K)^2')
                self.H_third_m = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 's__third_m', '-')
                self.HQTcov = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym' ,'s_qt_covar', '(J/kg/K)(kg/kg)' )
            elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
                self.Hvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym' ,'thetal_var', 'K^2')
                self.H_third_m = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'thetal_third_m', '-')
                self.HQTcov = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym' ,'thetal_qt_covar', 'K(kg/kg)' )

        return

    cpdef zero_tendencies(self):
        self.U.zero_tendencies(self.Gr)
        self.V.zero_tendencies(self.Gr)
        self.QT.zero_tendencies(self.Gr)
        self.H.zero_tendencies(self.Gr)
        return

    cpdef update(self,  TimeStepping TS):
        cdef:
            Py_ssize_t  k
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
                self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
                self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
                self.QT.values[k] +=  self.QT.tendencies[k] * TS.dt

        self.U.set_bcs(self.Gr)
        self.V.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)
        self.QT.set_bcs(self.Gr)

        if self.calc_tke:
            self.TKE.set_bcs(self.Gr)

        if self.calc_scalar_var:
            self.QTvar.set_bcs(self.Gr)
            self.Hvar.set_bcs(self.Gr)
            self.HQTcov.set_bcs(self.Gr)

        self.zero_tendencies()
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('u_mean')
        Stats.add_profile('v_mean')
        Stats.add_profile('qt_mean')
        Stats.add_profile('RH_mean')
        Stats.add_profile('dHdt_radiation')
        Stats.add_profile('dHdt_horz_adv')
        Stats.add_profile('dHdt_nudge')
        Stats.add_profile('dHdt_subsidence')
        Stats.add_profile('dQTdt_horz_adv')
        Stats.add_profile('dQTdt_nudge')
        Stats.add_profile('dQTdt_subsidence')
        if self.H.name == 's':
            Stats.add_profile('s_mean')
            Stats.add_profile('thetal_mean')
        elif self.H.name == 'thetal':
            Stats.add_profile('thetal_mean')

        Stats.add_profile('temperature_mean')
        Stats.add_profile('buoyancy_mean')
        Stats.add_profile('ql_mean')
        if self.calc_tke:
            Stats.add_profile('tke_mean')
        if self.calc_scalar_var:
            Stats.add_profile('Hvar_mean')
            Stats.add_profile('QTvar_mean')
            Stats.add_profile('HQTcov_mean')

            Stats.add_profile('W_third_m')
            Stats.add_profile('H_third_m')
            Stats.add_profile('QT_third_m')

        Stats.add_profile('cloud_fraction_mean')

        Stats.add_ts('lwp_mean')
        Stats.add_ts('cloud_base_mean')
        Stats.add_ts('cloud_top_mean')
        Stats.add_ts('cloud_cover_mean')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            double [:] arr = self.U.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw]
            Py_ssize_t k
        Stats.write_profile('u_mean', arr)
        Stats.write_profile('v_mean',self.V.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('qt_mean',self.QT.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('ql_mean',self.QL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('temperature_mean',self.T.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('RH_mean',self.RH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('buoyancy_mean',self.B.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dHdt_radiation',self.H.radiation[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dHdt_horz_adv',self.H.horz_adv[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dHdt_nudge',self.H.nudge[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dHdt_subsidence',self.H.subsidence[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dQTdt_horz_adv',self.QT.horz_adv[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dQTdt_nudge',self.QT.nudge[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('dQTdt_subsidence',self.QT.subsidence[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 's':
            Stats.write_profile('s_mean',self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('thetal_mean',self.THL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        elif self.H.name == 'thetal':
            Stats.write_profile('thetal_mean',self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_tke:
            Stats.write_profile('tke_mean',self.TKE.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('W_third_m',self.W_third_m.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_scalar_var:
            Stats.write_profile('Hvar_mean',self.Hvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('QTvar_mean',self.QTvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('HQTcov_mean',self.HQTcov.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

            Stats.write_profile('H_third_m',self.H_third_m.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('QT_third_m',self.QT_third_m.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('cloud_fraction_mean',self.cloud_fraction.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_ts('cloud_cover_mean', self.cloud_cover)

        self.mean_cloud_diagnostics()
        Stats.write_ts('lwp_mean', self.lwp)
        Stats.write_ts('cloud_base_mean',  self.cloud_base)
        Stats.write_ts('cloud_top_mean',   self.cloud_top)
        return

    cpdef mean_cloud_diagnostics(self):
        cdef Py_ssize_t k
        self.lwp = 0.
        self.cloud_base   = self.Gr.z_half[self.Gr.nzg - self.Gr.gw - 1]
        self.cloud_top    = 0.

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            self.lwp += self.Ref.rho0_half[k] * self.QL.values[k] * self.Gr.dz

            if self.QL.values[k] > 1e-8:
                self.cloud_base  = fmin(self.cloud_base,  self.Gr.z_half[k])
                self.cloud_top   = fmax(self.cloud_top,   self.Gr.z_half[k])
        return

    cpdef satadjust(self):
        cdef:
            Py_ssize_t k
            eos_struct sa
            double rho, qv, qt, h, p0

        with nogil:
            for k in xrange(self.Gr.nzg):
                h = self.H.values[k]
                qt = self.QT.values[k]
                p0 = self.Ref.p0_half[k]
                sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, p0, qt, h )
                self.QL.values[k] = sa.ql
                self.T.values[k] = sa.T
                qv = qt - sa.ql
                self.THL.values[k] = t_to_thetali_c(p0, sa.T, qt, sa.ql,0.0)
                rho = rho_c(p0, sa.T, qt, qv)
                self.B.values[k] = buoyancy_c(self.Ref.rho0_half[k], rho)
                self.RH.values[k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, self.T.values[k])

        return
