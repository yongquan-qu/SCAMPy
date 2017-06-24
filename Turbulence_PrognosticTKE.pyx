#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import pylab as plt
import numpy as np
include "parameters.pxi"
import cython
cimport  EDMF_Updrafts
from Grid cimport Grid
cimport EDMF_Environment
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport SurfaceBase
from Cases cimport  CasesBase
from ReferenceState cimport  ReferenceState
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport  *
from turbulence_functions cimport *
from utility_functions cimport interp2pt, gaussian_mean
from libc.math cimport fmax, sqrt, exp, pow, cbrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from Turbulence_BulkSteady cimport EDMF_BulkSteady



# Zhihong's scheme
cdef class EDMF_PrognosticTKE(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self, paramlist,  Gr, Ref)
        # Set the number of updrafts (1)
        try:
            self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        except:
            self.n_updrafts = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft')

        try:
            self.updraft_iterations = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_iterations']
        except:
            self.updraft_iterations = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft iteration')
        try:
            self.const_area = namelist['turbulence']['EDMF_PrognosticTKE']['constant_area']
        except:
            self.const_area = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to constant area fraction')
        try:
            self.use_local_micro = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        except:
            self.use_local_micro = True
        try:
            self.prognostic_rescale = namelist['turbulence']['EDMF_PrognosticTKE']['prognostic_rescale']
        except:
            self.prognostic_rescale = 1.0

        try:
            if namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z
            elif namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] == 'cloudy':
                self.entr_detr_fp = entr_detr_cloudy
            elif namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] == 'dry':
                self.entr_detr_fp = entr_detr_dry
            else:
                print('Turbulence--EDMF_PrognosticTKE: Entrainment rate namelist option is not recognized')
        except:
            self.entr_detr_fp = entr_detr_inverse_z
            print('Turbulence--EDMF_PrognosticTKE: defaulting to cloudy entrainment formulation')
        try:
            self.wu_option = namelist['turbulence']['EDMF_PrognosticTKE']['wu_option']
        except:
            self.wu_option = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to option 1 for updraft velocity numerics.')
        try:
            self.wu_min = namelist['turbulence']['EDMF_PrognosticTKE']['wu_min']
        except:
            self.wu_min = 0.0
            print('Turbulence--EDMF_PrognosticTKE: defaulting to 0 for updraft velocity minimum value.')
        try:
            self.max_area_factor = namelist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        except:
            self.max_area_factor = 1.0
            print('Turbulence--EDMF_PrognosticTKE: defaulting to unity ratio of max updraft fraction to surface updraft fraction')
        try:
            self.updraft_surface_height = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_surface_height']
        except:
            self.updraft_surface_height = 100.0
            print('Turbulence--EDMF_PrognosticTKE: defaulting to 100 m height for buoyant tail entrainment')
        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False


        self.extrapolate_buoyancy = True
        self.implicit_factor = 0.5

        # Get values from paramlist
        self.surface_area = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.surface_scalar_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_scalar_coeff']
        self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
        self.w_entr_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['w_entr_coeff']
        self.w_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['w_buoy_coeff']
        self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']


        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = EDMF_Updrafts.UpdraftMicrophysics(paramlist, self.n_updrafts, Gr, Ref)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EDMF_Environment.EnvironmentVariables(namelist,Gr)
        # Create the class for environment thermodynamics
        self.EnvThermo = EDMF_Environment.EnvironmentThermodynamics(namelist, Gr, Ref, self.EnvVar)

        # Entrainment rates
        self.entr_w = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_w = np.zeros((self.n_updrafts,Gr.nzg),dtype=np.double,order='c')
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # tke source terms
        self.tke_buoy = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_dissipation = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_entr_gain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_detr_loss = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_shear = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # Near-surface BC of updraft area fraction
        self.area_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.init_sc_upd= np.zeros((self.n_updrafts,),dtype=np.double, order='c')

       # Mass flux tendencies of mean scalars (for output)
        self.massflux_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')


       # (Eddy) diffusive tendencies of mean scalars (for output)
        self.diffusive_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')

       # Vertical fluxes for output
        self.massflux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_tke = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        return

    cpdef initialize(self, GridMeanVariables GMV):
        self.UpdVar.initialize(GMV)

        cdef:
            Py_ssize_t i,k
            double res_fac, gaussian_std
            double limpart_tot = 0.0, limpart_low = 0.0, limpart_upp = 0.0
            double lower_lim, upper_lim, init_a_u


        res_fac = gaussian_mean(0.9, 1.0)
        gaussian_std = res_fac

        for i in xrange(self.n_updrafts):
            limpart_tot += self.UpdVar.updraft_exponent ** i
        for i in xrange(self.n_updrafts):
            limpart_upp += self.UpdVar.updraft_exponent ** i
            lower_lim = 1.0 - self.UpdVar.updraft_fraction * (1.0 - limpart_low/limpart_tot)
            upper_lim = 1.0 - self.UpdVar.updraft_fraction * (1.0 - limpart_upp/limpart_tot)
            self.area_surface_bc[i]  = upper_lim - lower_lim
            res_fac = gaussian_mean(lower_lim, upper_lim)
            self.init_sc_upd[i] = res_fac/gaussian_std
            limpart_low = limpart_upp

        return

    # Initialize the IO pertaining to this class
    cpdef initialize_io(self, NetCDFIO_Stats Stats):

        self.UpdVar.initialize_io(Stats)
        self.EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_w')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('massflux')
        Stats.add_profile('massflux_h')
        Stats.add_profile('massflux_qt')
        Stats.add_profile('massflux_tendency_h')
        Stats.add_profile('massflux_tendency_qt')
        Stats.add_profile('diffusive_flux_h')
        Stats.add_profile('diffusive_flux_qt')
        Stats.add_profile('diffusive_tendency_h')
        Stats.add_profile('diffusive_tendency_qt')
        Stats.add_profile('total_flux_h')
        Stats.add_profile('total_flux_qt')
        Stats.add_profile('massflux_tke')
        Stats.add_profile('mixing_length')
        Stats.add_profile('tke_buoy')
        Stats.add_profile('tke_dissipation')
        Stats.add_profile('tke_entr_gain')
        Stats.add_profile('tke_detr_loss')
        Stats.add_profile('tke_shear')
        Stats.add_profile('updraft_qt_precip')
        Stats.add_profile('updraft_thetal_precip')


        return

    cpdef io(self, NetCDFIO_Stats Stats):

        cdef:
            Py_ssize_t k, i
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg-self.Gr.gw
            double [:] mean_entr_w = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_entr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_detr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        mean_entr_w[k] += self.UpdVar.Area.values[i,k] * self.entr_w[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_entr_sc[k] += self.UpdVar.Area.values[i,k] * self.entr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_detr_sc[k] += self.UpdVar.Area.values[i,k] * self.detr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]


        Stats.write_profile('entrainment_w', mean_entr_w[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('entrainment_sc', mean_entr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('detrainment_sc', mean_detr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux', np.sum(self.m[:,self.Gr.gw-1:self.Gr.nzg-self.Gr.gw -1], axis=0))
        Stats.write_profile('massflux_h', self.massflux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('massflux_qt', self.massflux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('massflux_tendency_h', self.massflux_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_qt', self.massflux_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_flux_h', self.diffusive_flux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('diffusive_flux_qt', self.diffusive_flux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('diffusive_tendency_h', self.diffusive_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_qt', self.diffusive_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('total_flux_h', np.add(self.massflux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1],
                                                   self.diffusive_flux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1]))
        Stats.write_profile('total_flux_qt', np.add(self.massflux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1],
                                                    self.diffusive_flux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1]))
        Stats.write_profile('massflux_tke', self.massflux_tke[kmin-1:kmax-1])
        Stats.write_profile('mixing_length', self.mixing_length[kmin:kmax])
        Stats.write_profile('tke_buoy', self.tke_buoy[kmin:kmax])
        Stats.write_profile('tke_dissipation', self.tke_dissipation[kmin:kmax])
        Stats.write_profile('tke_entr_gain', self.tke_entr_gain[kmin:kmax])
        Stats.write_profile('tke_detr_loss', self.tke_detr_loss[kmin:kmax])
        Stats.write_profile('tke_shear', self.tke_shear[kmin:kmax])
        Stats.write_profile('updraft_qt_precip', self.UpdMicro.prec_source_qt_tot[kmin:kmax])
        Stats.write_profile('updraft_thetal_precip', self.UpdMicro.prec_source_h_tot[kmin:kmax])

        return



    # Perform the update of the scheme
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS ):
        cdef:
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg - self.Gr.gw


        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        self.reset_surface_tke(GMV, Case)
        self.decompose_environment(GMV, 'values')

        if TS.nstep > 2:
            self.compute_prognostic_updrafts(GMV, Case, TS)
        else:
            self.compute_diagnostic_updrafts(GMV, Case)

        self.decompose_environment(GMV, 'values')
        self.update_GMV_MF(GMV, TS)
        self.decompose_environment(GMV, 'mf_update')
        self.EnvThermo.satadjust(self.EnvVar, GMV)

        # print('compute_eddy_diffusivities_tke')
        if TS.nstep > 1000000000:
            self.compute_eddy_diffusivities_tke(GMV, Case)
        else:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)

        self.update_GMV_ED(GMV, Case, TS)
        self.compute_tke(GMV, Case, TS)


        if TS.nstep > 1000000000000:
            # PLOTS
            plt.figure('Updated  W')
            plt.plot(self.UpdVar.W.values[0,kmin-1:kmax], self.Gr.z[kmin-1:kmax], label='upd')
            plt.plot(self.EnvVar.W.values[kmin:kmax],self.Gr.z_half[kmin:kmax], label='env')
            plt.grid()
            plt.legend(loc=0)

            plt.figure('Updated Updraft Area')
            plt.plot(self.UpdVar.Area.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax], label='upd')
            plt.grid()

            plt.figure('Updated Buoyancy')
            plt.plot(self.UpdVar.B.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax], label='upd')
            plt.plot(self.EnvVar.B.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='env')
            plt.grid()

            plt.figure('Updated QT')
            plt.plot(self.UpdVar.QT.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='upd')
            plt.plot(self.EnvVar.QT.values[kmin:kmax],self.Gr.z_half[kmin:kmax],label='env')
            plt.plot(GMV.QT.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
            plt.plot(GMV.QT.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
            plt.plot(GMV.QT.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
            plt.grid()
            plt.legend(loc=0)

            plt.figure('entr/detr')
            plt.plot(self.entr_w[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='entr_w')
            plt.plot(self.entr_sc[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='entr_sc')
            plt.plot(self.detr_w[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='detr_w')
            plt.plot(self.detr_sc[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='detr_sc')
            plt.legend(loc=0)
            plt.grid()

            plt.figure('Mixing length')
            plt.plot(self.mixing_length[kmin:kmax], self.Gr.z_half[kmin:kmax],label='KM')
            plt.grid()

            plt.figure('TKE Sources')
            plt.plot(self.tke_dissipation[kmin:kmax], self.Gr.z_half[kmin:kmax],label='dissip')
            plt.plot(self.tke_buoy[kmin:kmax], self.Gr.z_half[kmin:kmax],label='buoy')
            plt.plot(self.tke_shear[kmin:kmax], self.Gr.z_half[kmin:kmax],label='shear')
            plt.plot(self.tke_detr_loss[kmin:kmax], self.Gr.z_half[kmin:kmax],label='detr_loss')
            plt.plot(self.tke_entr_gain[kmin:kmax], self.Gr.z_half[kmin:kmax],label='entr_gain')
            plt.legend(loc=0)
            plt.grid()

            plt.figure('Viscosity')
            plt.plot(self.KM.values[kmin:kmax], self.Gr.z_half[kmin:kmax],label='KM')
            plt.plot(self.KH.values[kmin:kmax], self.Gr.z[kmin:kmax],label='KH')
            plt.grid()
            plt.legend(loc=0)

            plt.figure('Updated TKE')
            plt.plot(self.EnvVar.TKE.values[kmin:kmax],self.Gr.z_half[kmin:kmax],label='env')
            plt.plot(GMV.TKE.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
            plt.plot(GMV.TKE.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
            plt.plot(GMV.TKE.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
            plt.legend(loc=0)

            plt.figure('Updated H')
            plt.plot(self.UpdVar.H.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='upd')
            plt.plot(self.EnvVar.H.values[kmin:kmax], self.Gr.z_half[kmin:kmax],label='env')
            plt.plot(GMV.H.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
            plt.plot(GMV.H.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
            plt.plot(GMV.H.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
            plt.legend(loc=0)
            plt.grid()

        #     # plt.figure('Updated U')
        #     # plt.plot(GMV.U.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        #     # plt.plot(GMV.U.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        #     # plt.plot(GMV.U.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        #     # plt.legend(loc=0)
        #     #
        #     # plt.figure('Updated V')
        #     # plt.plot(GMV.V.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        #     # plt.plot(GMV.V.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        #     # plt.plot(GMV.V.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        #     # plt.legend(loc=0)
        #
            plt.show()



        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)
        return

    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        cdef:
            Py_ssize_t iter_

        self.UpdVar.set_new_with_values()
        self.UpdVar.set_old_with_values()
        self.set_updraft_surface_bc(GMV, Case)

        for iter_ in xrange(self.updraft_iterations):
            self.compute_entrainment_detrainment(GMV,Case)
            # self.solve_updraft_velocity(TS)
            # self.solve_area_fraction(GMV, TS)
            self.solve_updraft_velocity_area(GMV,TS)
            self.solve_updraft_scalars(GMV, Case, TS)
            self.UpdVar.set_values_with_new()
            self.UpdThermo.buoyancy(self.UpdVar, GMV, self.extrapolate_buoyancy)



        return

    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz
            double e_srf = 3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar
            eos_struct sa
            entr_struct ret
            double a,b,c, w2, w, w_mid, w_low, denom

        self.compute_entrainment_detrainment(GMV, Case)

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.values[i,gw] = (GMV.H.values[gw] + 0.3 * Case.Sur.rho_hflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])
                self.UpdVar.QT.values[i,gw] = (GMV.QT.values[gw] + 0.3 * Case.Sur.rho_qtflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])
                # Find the cloud liquid content
                sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[gw],
                         self.UpdVar.QT.values[i,gw], self.UpdVar.H.values[i,gw])
                self.UpdVar.QL.values[i,gw] = sa.ql
                self.UpdVar.T.values[i,gw] = sa.T
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    denom = 1.0 + self.entr_sc[i,k] * dz
                    self.UpdVar.H.values[i,k] = (self.UpdVar.H.values[i,k-1] + self.entr_sc[i,k] * dz * self.EnvVar.H.values[k])/denom
                    self.UpdVar.QT.values[i,k] = (self.UpdVar.QT.values[i,k-1] + self.entr_sc[i,k] * dz * self.EnvVar.QT.values[k])/denom

                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                    self.UpdVar.QL.values[i,k] = sa.ql
                    self.UpdVar.T.values[i,k] = sa.T

        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdThermo.buoyancy(self.UpdVar, GMV, self.extrapolate_buoyancy)

        # Solve updraft velocity equation
        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.W.values[i, self.Gr.gw-1] = 0.0  #self.w_surface_bc[i]
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    w = self.UpdVar.W.values[i,k-1]
                    w2 = (w * w + 2.0 * dz *(self.w_buoy_coeff *self.UpdVar.B.values[i,k]
                                             + self.w_entr_coeff *self.entr_w[i,k] * w * (self.EnvVar.W.values[k-1] - w)))
                    if w2 > 0.0:
                        self.UpdVar.W.values[i,k] = sqrt(w2)
                    else:
                        self.UpdVar.W.values[i,k:] = 0
                        break
        self.UpdVar.W.set_bcs(self.Gr)
        # solve_area_fraction
        # if self.const_area:
        #     with nogil:
        #         for i in xrange(self.n_updrafts):
        #             self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
        #             for k in xrange(gw+1, self.Gr.nzg):
        #                 if self.UpdVar.W.values[i,k] > 0.0:
        #                     self.UpdVar.Area.values[i,k] = self.area_surface_bc[i]
        #                 else:
        #                     # the updraft has terminated so set its area fraction to zero at this height and all heights above
        #                     self.UpdVar.Area.values[i,k] =  self.area_surface_bc[i]
        #                     self.UpdVar.H.values[i,k] = GMV.H.values[k]
        #                     self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
        #                     self.UpdVar.B.values[i,k] = 0.0
        #                     self.UpdVar.QL.values[i,k] = 0.0
        #                     self.UpdVar.T.values[i,k] = GMV.T.values[k]
        # else:
        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                w_mid = 0.5* (self.UpdVar.W.values[i,gw])
                for k in xrange(gw+1, self.Gr.nzg):
                    w_low = w_mid
                    w_mid = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                    if self.UpdVar.W.values[i,k] > 0.0:
                        self.UpdVar.Area.values[i,k] = (self.Ref.rho0_half[k-1]*self.UpdVar.Area.values[i,k-1]*w_low/
                                                        (1.0-(self.entr_sc[i,k]-self.detr_sc[i,k])*dz)/w_mid/self.Ref.rho0_half[k])
                        # # Limit the increase in updraft area when the updraft decelerates
                        if self.UpdVar.Area.values[i,k] >  self.max_area_factor * self.area_surface_bc[i]:
                            self.UpdVar.Area.values[i,k] = self.max_area_factor * self.area_surface_bc[i]
                    else:
                        # the updraft has terminated so set its area fraction to zero at this height and all heights above
                        self.UpdVar.Area.values[i,k] = 0.0
                        self.UpdVar.H.values[i,k] = GMV.H.values[k]
                        self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                        self.UpdVar.B.values[i,k] = 0.0
                        self.UpdVar.T.values[i,k] = GMV.T.values[k]
                        self.UpdVar.QL.values[i,k] = 0.0

        self.UpdVar.Area.set_bcs(self.Gr)

        return



    cpdef compute_tke(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double ws= self.wstar, us = Case.Sur.ustar, zs = self.zi, z

        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        self.compute_mixing_length(Case.Sur.obukhov_length)
        if TS.nstep > 3:
            self.compute_tke_buoy(GMV)
            self.compute_tke_dissipation(TS)
            self.compute_tke_entr_detr()
            self.compute_tke_shear(GMV)

            self.update_tke_MF(GMV, TS)
            self.reset_surface_tke(GMV, Case)
            self.update_tke_ED(GMV, Case, TS)
            self.reset_surface_tke(GMV, Case)
        else:
            with nogil:
                for k in xrange(self.Gr.nzg):
                    z = self.Gr.z_half[k]
                    GMV.TKE.values[k] = ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
                    GMV.TKE.new[k] = GMV.TKE.values[k]
                    GMV.TKE.mf_update[k] = GMV.TKE.values[k]

        return

    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    cpdef compute_mixing_length(self, double obukhov_length):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double tau = 0.5 * self.zi / self.wstar
            double l1, l2, z_

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                l1 = tau * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                z_ = self.Gr.z_half[k]
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_
                self.mixing_length[k] = 1.0/(1.0/fmax(l1,1e-10) + 1.0/l2)
        return


    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double lm

        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)
        else:
            self.compute_mixing_length(Case.Sur.obukhov_length)
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    lm = self.mixing_length[k]
                    self.KM.values[k] = self.tke_ed_coeff * lm * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                    self.KH.values[k] = self.KM.values[k] / self.prandtl_number

        return

    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case):

        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        cdef:
            Py_ssize_t i
            double e_srf = 3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar
            Py_ssize_t gw = self.Gr.gw

        with nogil:
            for i in xrange(self.n_updrafts):
                #area: set in initialize
                self.w_surface_bc[i] = 0.0
                self.h_surface_bc[i] = (GMV.H.values[gw] + self.surface_scalar_coeff * self.init_sc_upd[i]
                                        * Case.Sur.rho_hflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])
                self.qt_surface_bc[i] = (GMV.QT.values[gw] + self.surface_scalar_coeff * self.init_sc_upd[i]
                                         * Case.Sur.rho_qtflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])
                # self.h_surface_bc[i] = GMV.H.values[gw]
                # self.qt_surface_bc[i] = GMV.QT.values[gw]

        return



    cpdef reset_surface_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef double tke_surface = 3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar

        GMV.TKE.values[self.Gr.gw] = tke_surface
        GMV.TKE.mf_update[self.Gr.gw] = tke_surface
        return



    # Find values of environmental variables by subtracting updraft values from grid mean values
    # whichvals used to check which substep we are on--correspondingly use 'GMV.SomeVar.value' (last timestep value)
    # or GMV.SomeVar.mf_update (GMV value following massflux substep)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals):

        # first make sure the 'bulkvalues' of the updraft variables are updated
        self.UpdVar.set_means(GMV)

        cdef:
            Py_ssize_t k, gw = self.Gr.gw
            double val1, val2, au_full


        if whichvals == 'values':

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1
                    self.EnvVar.QT.values[k] = val1 * GMV.QT.values[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.values[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W--interpolate area fraction to the "full" grid points
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

                    # Zhihong's comment: "unsure about TKE decomposition"
                    self.EnvVar.TKE.values[k] =  val1 * GMV.TKE.values[k]

        elif whichvals == 'mf_update':
            # same as above but replace GMV.SomeVar.values with GMV.SomeVar.mf_update

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1

                    self.EnvVar.QT.values[k] = val1 * GMV.QT.mf_update[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.mf_update[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]
                    # Zhihong's comment: "unsure about TKE decomposition"
                    self.EnvVar.TKE.values[k] =  val1 * GMV.TKE.mf_update[k]

        return

    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            entr_struct ret
        self.update_inversion(GMV, Case.inversion_option)
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    ret = self.entr_detr_fp(self.Gr.z[k], self.Gr.z_half[k], self.zi)
                    self.entr_w[i,k] = ret.entr_w
                    self.entr_sc[i,k] = ret.entr_sc
                    self.detr_w[i,k] = ret.detr_w
                    self.detr_sc[i,k] = ret.detr_sc

        return


    # solve the updraft velocity equation
    cpdef solve_updraft_velocity(self,TimeStepping TS):
        cdef:
            double b1 = self.w_entr_coeff
            double b2 = self.w_buoy_coeff
            Py_ssize_t k, i
            double dz = self.Gr.dz
            double wa, wb2, wbl

        if self.wu_option == 1:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.W.new[i,self.Gr.gw-1] =  self.w_surface_bc[i]
                    for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        #Limit wa to be positive (which may not hold if w_env >> 0, or more specifically, w_env >~ dz/dt)
                        wa = fmax(dz*(TS.dti * self.prognostic_rescale
                                      - b1*self.EnvVar.W.values[k]*self.entr_w[i,k]), 0.0)
                        wb2 = ((1.0 + 2.0 * b1 * self.entr_w[i,k]*dz) *
                               (self.UpdVar.W.new[i,k-1]**2.
                                + 2.0 * dz * TS.dti * self.UpdVar.W.values[i,k] * self.prognostic_rescale
                               + 2.0 * b2 * interp2pt(self.UpdVar.B.values[i,k],self.UpdVar.B.values[i,k+1])*dz))

                        wb2 = fmax(wb2, 0.0)
                        self.UpdVar.W.new[i,k] = (-wa + sqrt(wa*wa + wb2))/(1.0 + 2.0 * b1 * self.entr_w[i,k]*dz)
                        if self.UpdVar.W.new[i,k] <= 0.0:
                            # self.UpdVar.W.new[i,k] = 0.0
                            self.UpdVar.W.new[i,k:] = 0.0
                            break

        elif self.wu_option == 2.0:
             with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.W.new[i,self.Gr.gw-1] = self.w_surface_bc[i]
                    for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        wa = fmax(dz * TS.dti, 0.0)
                        if self.entr_w[i,k] > 0.0:
                            wbl = (b2 * self.UpdVar.B.values[i,k] / b1 / self.entr_w[i,k]
                                   * (1.0 - exp(-2.0 * b1 * self.entr_w[i,k] * dz)))
                        else:
                            wbl = 2.0 * b2 * self.UpdVar.B.values[i,k] * dz
                        wb2 =  self.UpdVar.W.new[i,k-1]**2.* exp(-2.0 * b1 * self.entr_w[i,k] * dz) + 2.0 * dz * TS.dti * self.UpdVar.W.values[i,k] + wbl

                        wb2 = fmax(wb2, 0.0)
                        self.UpdVar.W.new[i,k] = -wa + sqrt(wa*wa + wb2)
                        if self.UpdVar.W.new[i,k] <= 0.0:
                            # self.UpdVar.W.new[i,k] = 0.0
                            self.UpdVar.W.new[i,k:] = 0.0
                            break


        self.UpdVar.W.set_bcs(self.Gr)

        return

    cpdef solve_area_fraction(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t gw = self.Gr.gw
            double au_lim
            double dti_pr = TS.dti * self.prognostic_rescale
            double dzi = self.Gr.dzi
            double imf = self.implicit_factor
            double exf = 1.0 - self.implicit_factor
            double wnew_k, w_k, wnew_km, w_km, rhof
            double c1, c2, c3, c4, c0

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.Area.new[i,gw] = self.area_surface_bc[i]
                au_lim = self.area_surface_bc[i] * self.max_area_factor
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    wnew_k = interp2pt(self.UpdVar.W.new[i,k-1], self.UpdVar.W.new[i,k])
                    if wnew_k > 0.0:
                        w_k = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                        wnew_km = interp2pt(self.UpdVar.W.new[i,k-2], self.UpdVar.W.new[i,k-1])
                        w_km = interp2pt(self.UpdVar.W.values[i,k-2], self.UpdVar.W.values[i,k-1])
                        rhof = self.Ref.rho0_half[k-1]/self.Ref.rho0_half[k]
                        c1 = dti_pr + imf * wnew_k * (dzi - self.entr_sc[i,k] + self.detr_sc[i,k])
                        c2 = dti_pr + exf * w_k * (self.entr_sc[i,k] - self.detr_sc[i,k] - dzi)
                        c3 = rhof * imf * wnew_km * dzi
                        c4 = rhof * exf * w_km * dzi
                        self.UpdVar.Area.new[i,k] = (c2/c1 * self.UpdVar.Area.values[i,k]
                                                     + c3/c1 * self.UpdVar.Area.new[i,k-1]
                                                     + c4/c1 * self.UpdVar.Area.values[i,k-1])
                        if self.UpdVar.Area.new[i,k] > au_lim:
                            c0 = imf * wnew_k * au_lim + exf * w_k * self.UpdVar.Area.values[i,k]
                            c1 = dti_pr + imf * wnew_k * (dzi - self.entr_sc[i,k] )
                            c2 = dti_pr + exf * w_k * (self.entr_sc[i,k] - dzi)
                            self.detr_sc[i,k] = (-c1/c0 * au_lim + c2/c0 * self.UpdVar.Area.values[i,k]
                                                     + c3/c0 * self.UpdVar.Area.new[i,k-1]
                                                     + c4/c0 * self.UpdVar.Area.values[i,k-1] )
                            self.UpdVar.Area.new[i,k] = au_lim
                    else:
                        self.UpdVar.Area.new[i,k] = 0.0
                    # with gil:
                    #     print(k, wnew_k,self.UpdVar.W.new[i,k], self.UpdVar.Area.new[i,k])

        self.UpdVar.Area.set_bcs(self.Gr)

        return

    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dzi = self.Gr.dzi
            double dti_ = TS.dti * self.prognostic_rescale
            double rho_ratio, whalf_kp, whalf_k
            double a1, a2 # groupings of terms in area fraction discrete equation
            double au_lim
            double anew_k, a_k, a_km, entr_w, detr_w, B_k
            double adv, buoy, exch, press # groupings of terms in velocity discrete equation
        #
        # plt.figure('entr')
        # plt.plot(self.entr_sc[0,gw:self.Gr.nzg-gw], self.Gr.z_half[gw:self.Gr.nzg-gw], label = 'begin entr')
        # plt.plot(self.detr_sc[0,gw:self.Gr.nzg-gw], self.Gr.z_half[gw:self.Gr.nzg-gw], label = 'begin detr')
        #


        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.n_updrafts):
                    self.entr_sc[i,gw] = 2.0 * dzi
                    self.detr_sc[i,gw] = 0.0
                    self.UpdVar.W.new[i,gw-1] = self.w_surface_bc[i]
                    self.UpdVar.Area.new[i,gw] = self.area_surface_bc[i]
                    au_lim = self.area_surface_bc[i] * self.max_area_factor
                    for k in range(gw, self.Gr.nzg-gw):

                        # First solve for updated area fraction at k+1
                        rho_ratio = self.Ref.rho0_half[k]/self.Ref.rho0_half[k+1]
                        whalf_kp = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k+1])
                        whalf_k = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                        # with gil:
                        #     print(self.entr_sc[i,k+1], self.detr_sc[i,k+1])
                        a1 = (dti_ + whalf_kp * (self.entr_sc[i,k+1] - self.detr_sc[i,k+1] - dzi))/dti_
                        a2 = whalf_k * dzi/dti_
                        self.UpdVar.Area.new[i,k+1] = a1 * self.UpdVar.Area.values[i,k+1] + a2 * self.UpdVar.Area.values[i,k]
                        # with gil:
                        #     print(k, self.UpdVar.Area.new[i,k+1], self.detr_sc[i,k+1])
                        if self.UpdVar.Area.new[i,k+1] > au_lim:
                            # with gil:
                            #     print('exceeded area lim', k, self.UpdVar.Area.values[i,k+1], self.detr_sc[i,k+1])
                            # a1 = 1.0 + whalf_kp/dti_ * (self.entr_sc[i,k+1] - dzi)
                            if whalf_kp * self.UpdVar.Area.values[i,k+1]  > 0.0:

                                self.detr_sc[i,k+1] = (dti_/whalf_kp / self.UpdVar.Area.values[i,k+1]
                                                       * (a1 * self.UpdVar.Area.values[i,k+1]
                                                          + a2 * self.UpdVar.Area.values[i,k] - au_lim))

                                self.UpdVar.Area.new[i,k+1]= au_lim
                            else:
                                self.UpdVar.Area.new[i,k+1]= au_lim
                                self.detr_sc[i,k+1] = self.detr_sc[i,k]
                            # with gil:
                            #     print('corrected', k, self.UpdVar.Area.values[i,k+1], self.detr_sc[i,k+1])

                        # Now solve for updraft velocity at k
                        rho_ratio = self.Ref.rho0[k-1]/self.Ref.rho0[k]
                        anew_k = interp2pt(self.UpdVar.Area.new[i,k], self.UpdVar.Area.new[i,k+1])
                        if anew_k > 1e-5:
                            a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                            a_km = interp2pt(self.UpdVar.Area.values[i,k-1], self.UpdVar.Area.values[i,k])
                            entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                            detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                            B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                            adv = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * self.UpdVar.W.values[i,k] * dzi
                                   - self.Ref.rho0[k-1] * a_km * self.UpdVar.W.values[i,k-1] * self.UpdVar.W.values[i,k-1] * dzi)
                            exch = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k]
                                    * (entr_w * self.EnvVar.W.values[k] - detr_w * self.UpdVar.W.values[i,k] ))
                            buoy = self.Ref.rho0[k] * a_k * B_k
                            press = 0.0*(self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * (entr_w + detr_w)
                                     * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k]))
                            self.UpdVar.W.new[i,k] = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * dti_
                                                      -adv + exch + buoy -press)/(self.Ref.rho0[k] * anew_k * dti_)


                            if self.UpdVar.W.new[i,k] <= 0.0:
                                self.UpdVar.W.new[i,k:] = 0.0
                                self.UpdVar.Area.new[i,k+2:] = 0.0
                                break
                        else:
                            self.UpdVar.W.new[i,k:] = 0.0
                            self.UpdVar.Area.new[i,k+2:] = 0.0
                            break
        # plt.plot(self.entr_sc[0,gw:self.Gr.nzg-gw], self.Gr.z_half[gw:self.Gr.nzg-gw], '--',label = 'end entr')
        # plt.plot(self.detr_sc[0,gw:self.Gr.nzg-gw], self.Gr.z_half[gw:self.Gr.nzg-gw],'--',  label = 'end detr')
        # plt.legend(loc=0)
        # plt.show()

        return



    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            double dz = self.Gr.dz
            Py_ssize_t gw = self.Gr.gw
            double dH_entr, dQT_entr, H_entr, QT_entr
            double c1, c2, c3, ct
            eos_struct sa
            double sqrt_e_srf =sqrt(3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar)


        if self.use_local_micro:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.H.new[i,gw] = self.h_surface_bc[i]
                    self.UpdVar.QT.new[i,gw]  = self.qt_surface_bc[i]
                    dH_entr = 1.8 * Case.Sur.rho_hflux/sqrt_e_srf * self.init_sc_upd[i] * self.Ref.alpha0_half[gw]
                    dQT_entr = 1.8 * Case.Sur.rho_qtflux/sqrt_e_srf * self.init_sc_upd[i] * self.Ref.alpha0_half[gw]
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp,
                             self.Ref.p0_half[gw], self.UpdVar.QT.new[i,gw], self.UpdVar.H.new[i,gw])
                    self.UpdVar.QL.new[i,gw] = sa.ql
                    self.UpdVar.T.new[i,gw] = sa.T
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[gw], self.UpdVar.T.new[i,gw],
                                                                       &self.UpdVar.QT.new[i,gw], &self.UpdVar.QL.new[i,gw],
                                                                       &self.UpdVar.H.new[i,gw], i, gw)
                    for k in xrange(gw+1, self.Gr.nzg-gw):
                        if k ==gw or self.Gr.z_half[k] < self.updraft_surface_height:
                            H_entr = self.EnvVar.H.values[k] + dH_entr
                            QT_entr = self.EnvVar.QT.values[k] + dQT_entr
                        else:
                            H_entr = self.EnvVar.H.values[k]
                            QT_entr = self.EnvVar.QT.values[k]

                        c1 = self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * dz * TS.dti * self.prognostic_rescale
                        c2 = self.Ref.rho0_half[k-1]  * self.UpdVar.Area.new[i,k-1] * interp2pt(self.UpdVar.W.new[i,k-1],self.UpdVar.W.new[i,k-2])
                        c3 = self.entr_sc[i,k] * self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * interp2pt(self.UpdVar.W.new[i,k-1],self.UpdVar.W.new[i,k])
                        ct = c1 + c2 + c3
                        if ct > 0.0:
                            c1 = c1/ct
                            c2 = c2/ct
                            c3 = c3/ct
                            self.UpdVar.H.new[i,k] = c1 * self.UpdVar.H.values[i,k] + c2 * self.UpdVar.H.new[i,k-1] + c3 * H_entr
                            self.UpdVar.QT.new[i,k] = c1 * self.UpdVar.QT.values[i,k] + c2 * self.UpdVar.QT.new[i,k-1] + c3 * QT_entr
                        else:
                            self.UpdVar.H.new[i,k] = self.EnvVar.H.values[k]
                            self.UpdVar.QT.new[i,k] = self.EnvVar.QT.values[k]
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.new[i,k], self.UpdVar.H.new[i,k])
                        self.UpdVar.QL.new[i,k] = sa.ql
                        self.UpdVar.T.new[i,k] = sa.T
                        self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[k], self.UpdVar.T.new[i,k],
                                                                       &self.UpdVar.QT.new[i,k], &self.UpdVar.QL.new[i,k],
                                                                       &self.UpdVar.H.new[i,k], i, k)
        else:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.H.new[i,gw] = self.h_surface_bc[i]
                    self.UpdVar.QT.new[i,gw]  = self.qt_surface_bc[i]

                    dH_entr = 1.8 * Case.Sur.rho_hflux/sqrt_e_srf * self.init_sc_upd[i] * self.Ref.alpha0_half[gw]
                    dQT_entr = 1.8 * Case.Sur.rho_qtflux/sqrt_e_srf * self.init_sc_upd[i] * self.Ref.alpha0_half[gw]
                    for k in xrange(gw+1, self.Gr.nzg-gw):
                        if k ==gw or self.Gr.z_half[k] < self.updraft_surface_height:
                            H_entr = self.EnvVar.H.values[k] + dH_entr
                            QT_entr = self.EnvVar.QT.values[k] + dQT_entr
                        else:
                            H_entr = self.EnvVar.H.values[k]
                            QT_entr = self.EnvVar.QT.values[k]

                        c1 = self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * dz * TS.dti
                        c2 = self.Ref.rho0_half[k-1]  * self.UpdVar.Area.new[i,k-1] * interp2pt(self.UpdVar.W.new[i,k-1],self.UpdVar.W.new[i,k-2])
                        c3 = self.entr_sc[i,k] * self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * interp2pt(self.UpdVar.W.new[i,k-1],self.UpdVar.W.new[i,k])

                        ct = c1 + c2 + c3
                        if ct > 0.0:
                            c1 = c1/ct
                            c2 = c2/ct
                            c3 = c3/ct
                            self.UpdVar.H.new[i,k] = c1 * self.UpdVar.H.values[i,k] + c2 * self.UpdVar.H.new[i,k-1] + c3 * H_entr
                            self.UpdVar.QT.new[i,k] = c1 * self.UpdVar.QT.values[i,k] + c2 * self.UpdVar.QT.new[i,k-1] + c3 * QT_entr
                        else:
                            self.UpdVar.H.new[i,k] = self.EnvVar.H.values[k]
                            self.UpdVar.QT.new[i,k] = self.EnvVar.QT.values[k]
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.new[i,k], self.UpdVar.H.new[i,k])
                        self.UpdVar.QL.new[i,k] = sa.ql
                        self.UpdVar.T.new[i,k] = sa.T
            self.UpdMicro.compute_sources(self.UpdVar)
            self.UpdMicro.update_updraftvars(self.UpdVar)

        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
        return



    # After updating the updraft variables themselves:
    # 1. compute the mass fluxes (currently not stored as class members, probably will want to do this
    # for output purposes)
    # 2. Apply mass flux tendencies and updraft microphysical tendencies to GMV.SomeVar.Values (old time step values)
    # thereby updating to GMV.SomeVar.mf_update
    # mass flux tendency is computed as 1st order upwind

    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t gw = self.Gr.gw
            double mf_tend_h=0.0, mf_tend_qt=0.0
            double env_h_interp, env_qt_interp
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        with nogil:
            for i in xrange(self.n_updrafts):
                self.m[i,gw-1] = 0.0
                for k in xrange(self.Gr.gw, self.Gr.nzg-1):
                    self.m[i,k] = ((self.UpdVar.W.values[i,k] - self.EnvVar.W.values[k] )* self.Ref.rho0[k]
                                   * interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1]))

        self.massflux_h[gw-1] = 0.0
        self.massflux_qt[gw-1] = 0.0
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                self.massflux_h[k] = 0.0
                self.massflux_qt[k] = 0.0
                env_h_interp = interp2pt(self.EnvVar.H.values[k], self.EnvVar.H.values[k+1])
                env_qt_interp = interp2pt(self.EnvVar.QT.values[k], self.EnvVar.QT.values[k+1])
                for i in xrange(self.n_updrafts):
                    self.massflux_h[k] += self.m[i,k] * (interp2pt(self.UpdVar.H.values[i,k],
                                                                   self.UpdVar.H.values[i,k+1]) - env_h_interp )
                    self.massflux_qt[k] += self.m[i,k] * (interp2pt(self.UpdVar.QT.values[i,k],
                                                                    self.UpdVar.QT.values[i,k+1]) - env_qt_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variables
        with nogil:

            for k in xrange(self.Gr.gw, self.Gr.nzg):
                mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
                mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdMicro.prec_source_h_tot[k]
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k]

                #No mass flux tendency for U, V
                GMV.U.mf_update[k] = GMV.U.values[k]
                GMV.V.mf_update[k] = GMV.V.values[k]
                GMV.TKE.mf_update[k] = GMV.TKE.values[k]

                # Prepare the output
                self.massflux_tendency_h[k] = mf_tend_h
                self.massflux_tendency_qt[k] = mf_tend_qt
                # self.massflux_h[k] = self.massflux_h[k] * self.Ref.alpha0_half[k]
                # self.massflux_qt[k] = self.massflux_qt[k] * self.Ref.alpha0_half[k]
        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

        GMV.satadjust()

        return






    # Update the grid mean variables with the tendency due to eddy diffusion
    # Km and Kh have already been updated
    # 2nd order finite differences plus implicit time step allows solution with tridiagonal matrix solver
    # Update from GMV.SomeVar.mf_update to GMV.SomeVar.new
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, self.Gr.dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] =  self.EnvVar.QT.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.QT.new[k+gw] = GMV.QT.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])
                self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
                self.diffusive_flux_qt[k+gw] = -rho_ae_K_m[k+gw] * (GMV.QT.new[k+gw+1] - GMV.QT.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            self.diffusive_flux_qt[gw-1] = Case.Sur.rho_qtflux*self.Ref.alpha0_half[gw]

        # Solve H
        with nogil:
            for k in xrange(nz):
                x[k] = self.EnvVar.H.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.H.new[k+gw] = GMV.H.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])
                self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
                self.diffusive_flux_h[k+gw] =  -rho_ae_K_m[k+gw] * (GMV.H.new[k+gw+1] - GMV.H.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]

        # Solve U
        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, self.Gr.dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.U.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_uflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.U.new[k+gw] = x[k]

        # Solve V
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.V.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_vflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.V.new[k+gw] = x[k]

        GMV.QT.set_bcs(self.Gr)
        GMV.H.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

        return



    cpdef compute_tke_buoy(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double db_dthl_d, db_dqt_d, db_dthl_s, db_dqt_s
            double qt_d, thl_d, qs_s, lh
            double theta_rho_mean, cf
            double grad_thl_minus=0.0, grad_qt_minus=0.0, grad_thl_plus=0, grad_qt_plus=0
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        # Note that source terms at the gw grid point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                grad_thl_minus = grad_thl_plus
                grad_qt_minus = grad_qt_plus
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus = (self.EnvVar.QT.values[k+1] - self.EnvVar.QT.values[k]) * self.Gr.dzi
                cf = self.EnvVar.CF.values[k]
                theta_rho_mean = theta_rho_c(self.Ref.p0_half[k], GMV.T.values[k],
                                             GMV.QT.values[k], GMV.QT.values[k]-GMV.QL.values[k])
                qt_d = (self.EnvVar.QT.values[k] - cf * self.EnvThermo.qt_cloudy[k])/ fmax((1.0 - cf),1.0e-10)
                thl_d = (self.EnvVar.THL.values[k] - cf * self.EnvThermo.thl_cloudy[k])/ fmax((1.0 - cf),1.0e-10)
                qs_s = self.EnvThermo.qt_cloudy[k] - self.EnvVar.QL.values[k]/fmax(cf, 1e-10)

                db_dthl_d = g/theta_rho_mean * (1.0 + (eps_vi-1.0) * qt_d)
                db_dqt_d = g/theta_rho_mean * (eps_vi - 1.0) * thl_d
                lh = latent_heat(self.EnvThermo.t_cloudy[k])

                db_dthl_s = g/theta_rho_mean * (1.0 + eps_vi * (1.0 +lh/Rv/self.EnvThermo.t_cloudy[k])
                                                * qs_s - self.EnvThermo.qt_cloudy[k])
                db_dthl_s /= (1.0 + lh * lh/(cpm_c(self.EnvThermo.qt_cloudy[k]) * Rv * self.EnvThermo.t_cloudy[k]
                                             * self.EnvThermo.t_cloudy[k]) * qs_s)
                db_dqt_s = (lh/cpm_c(self.EnvThermo.qt_cloudy[k])/self.EnvThermo.t_cloudy[k] * db_dthl_s
                            - g/theta_rho_mean) * self.EnvThermo.thl_cloudy[k]

                self.tke_buoy[k] = ( -self.KH.values[k] *interp2pt(grad_thl_plus, grad_thl_minus)
                                      * ((1.0 - cf)*db_dthl_d + cf * db_dthl_s)
                                     - self.KH.values[k] * interp2pt(grad_qt_plus, grad_qt_minus)
                                      *  ((1.0 - cf) * db_dqt_d + cf * db_dqt_s)) * ae[k]
        return

    # Note we need mixing length again here....
    cpdef compute_tke_dissipation(self, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
            #tke diss coeff

        # does mixing length need to be recomputed? Even if no change in GMV.TKE, if updraft area changes
        # this would change environmental tke (assuming it is still adjusted based on tke)
        # first pass...assume we can re-use
        # Using the "semi-implicit formulation" with dissipation averaged over timestep
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_dissipation[k] = ae[k] * (self.EnvVar.TKE.values[k] * TS.dti  *
                                           (-1.0 + pow((1.0 + 0.5 * self.tke_diss_coeff * TS.dt *
                                                        sqrt(fmax(self.EnvVar.TKE.values[k],0.0))/fmax(self.mixing_length[k],1e-5)),-2.0)))

                # self.tke_dissipation[k] = -ae[k] * pow(fmax(self.EnvVar.TKE.values[k],0), 1.5)/fmax(self.mixing_length[k],1e-6)
        return

    # Note updrafts' entrainment rate = environment's detrainment rate
    # and updrafts' detrainment rate = environment's entrainment rate
    # Here we use the terminology entrainment/detrainment relative to the __environment__
    cpdef compute_tke_entr_detr(self):
        cdef:
            Py_ssize_t i, k
            double [:] detr_tot_env = np.sum(np.multiply(self.entr_w, self.m), axis=0.0) # At W points
            # double [:,:] m_entr_env = np.multiply(self.m, self.detr_w)
            double w_u, w_e


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_detr_loss[k] = -interp2pt(detr_tot_env[k-1], detr_tot_env[k]) * self.EnvVar.TKE.values[k]
                self.tke_entr_gain[k] = 0.0
                for i in xrange(self.n_updrafts):
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    w_e = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                    # self.tke_entr_gain[k] += 0.5 * interp2pt(m_entr_env[i,k-1],m_entr_env[i,k]) * (w_u - w_e) * (w_u - w_e)
                    self.tke_entr_gain[k] +=( 0.5 * interp2pt(self.m[i,k-1],self.m[i,k])
                                              * interp2pt(self.detr_w[i,k-1], self.detr_w[i,k]) * (w_u - w_e) * (w_u - w_e))
                self.tke_detr_loss[k] *= self.Ref.alpha0_half[k]
                self.tke_entr_gain[k] *= self.Ref.alpha0_half[k]
        return

    cpdef compute_tke_shear(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double du_high = 0.0
            double dv_high = 0.0
            double dw_high = 2.0 * self.EnvVar.W.values[gw]  * self.Gr.dzi
            double du_low, dv_low, dw_low,

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                du_low = du_high
                dv_low = dv_high
                dw_low = dw_high
                du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi
                dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi
                dw_high = (self.EnvVar.W.values[k+1] - self.EnvVar.W.values[k]) * self.Gr.dzi
                self.tke_shear[k] =( ae[k] * self.KM.values[k] *
                                    ( pow(interp2pt(du_low, du_high),2.0) +  pow(interp2pt(dv_low, dv_high),2.0)
                                      + pow(interp2pt(dw_low, dw_high),2.0)))
        return

    # note: may want to switch to env_tke_interp = self.EnvVar.TKE.values[k], i.e. purely upwind treatment
    cpdef update_tke_MF(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] m_total = np.sum(self.m, axis=0)
            double env_tke_interp, mf_tend_tke, val1, val2

        self.massflux_tke[gw-1] = 0.0

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                env_tke_interp = interp2pt(self.EnvVar.TKE.values[k], self.EnvVar.TKE.values[k+1])
                self.massflux_tke[k] = m_total[k] * (- env_tke_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variable
        with nogil:

            for k in xrange(self.Gr.gw+1, self.Gr.nzg):
                mf_tend_tke = -(self.massflux_tke[k] - self.massflux_tke[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
                GMV.TKE.mf_update[k] =fmax( GMV.TKE.values[k] +  TS.dt * (mf_tend_tke  + self.tke_buoy[k]
                                                                          + self.tke_dissipation[k] + self.tke_entr_gain[k]
                                                                          + self.tke_detr_loss[k] + self.tke_shear[k]), 0.0)

            for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1
                    # Zhihong's comment: "unsure about TKE decomposition"
                    self.EnvVar.TKE.values[k] =  val1 * GMV.TKE.mf_update[k]



        GMV.TKE.set_bcs(self.Gr)


        return

    cpdef update_tke_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion_dirichlet(nzg, gw, self.Gr.dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] =  self.EnvVar.TKE.values[k+gw]
            # x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.TKE.new[k+gw] = GMV.TKE.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.TKE.values[k+gw])
            #     self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
            #     self.diffusive_flux_qt[k+gw] = -rho_ae_K_m[k+gw] * (GMV.QT.new[k+gw+1] - GMV.QT.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            # self.diffusive_flux_qt[gw-1] = Case.Sur.rho_qtflux*self.Ref.alpha0_half[gw]
        return
