#!python
# cython: boundscheck=False
# cython: wraparound=True
# cython: initializedcheck=False
# cython: cdivision=True

#Adapated from PyCLES: https://github.com/pressel/pycles

from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
cimport numpy as np
import numpy as np
import netCDF4 as nc
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from thermodynamic_functions cimport t_to_entropy_c, eos_first_guess_entropy, eos, alpha_c
include 'parameters.pxi'


cdef class ReferenceState:
    def __init__(self, Grid Gr ):

        self.p0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.p0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        return




    def initialize(self, Grid Gr, NetCDFIO_Stats Stats, namelist):
        '''
        Initilize the reference profiles. The function is typically called from the case
        specific initialization fucntion defined in Initialization.pyx
        :param Gr: Grid class
        :param Thermodynamics: Thermodynamics class
        :param NS: StatsIO class
        :param Pa:  ParallelMPI class
        :return:
        '''

        cdef:
            double s
            double[:] p_
            double[:] p_half_
            double[:] temperature = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] temperature_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] alpha = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] alpha_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

            double[:] ql = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] qi = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] qv = np.zeros(Gr.nzg, dtype=np.double, order='c')

            double[:] ql_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] qi_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
            double[:] qv_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        if namelist['meta']['casename'] == 'LES_driven_SCM':
            les_data = nc.Dataset(Gr.les_filename,'r')
            les_alpha_half = np.array(les_data.groups['reference'].variables['alpha0'])
            les_alpha = np.array(les_data.groups['reference'].variables['alpha0_full'])
            les_p = np.array(les_data.groups['reference'].variables['p0_full'])
            les_p_half = np.array(les_data.groups['reference'].variables['p0'])
            z_les = np.array(les_data.groups['profiles'].variables['z'])
            z_les_half = np.array(les_data.groups['profiles'].variables['z_half'])
            f_les_alpha_half = interp1d(z_les_half, les_alpha_half, fill_value="extrapolate")
            f_les_p_half = interp1d(z_les_half, les_p_half, fill_value="extrapolate")
            f_les_alpha = interp1d(z_les, les_alpha, fill_value="extrapolate")
            f_les_p = interp1d(z_les, les_p, fill_value="extrapolate")
            alpha_half = f_les_alpha_half(Gr.z_half)
            alpha = f_les_alpha(Gr.z)
            p_half = f_les_p_half(Gr.z_half)
            p_ = f_les_p(Gr.z_half)

            self.alpha0_half = alpha_half
            self.alpha0 = alpha
            self.p0 = p_
            self.p0_half = p_half
            self.rho0 = 1.0 / np.array(self.alpha0)
            self.rho0_half = 1.0 / np.array(self.alpha0_half)

            self.Pg = les_p[0]
            self.qtg = np.array(les_data.groups['profiles'].variables['qt_mean'])[0,0]
            self.Tg = np.array(les_data.groups['timeseries'].variables['surface_temperature'])[0]
        else:
            self.sg = t_to_entropy_c(self.Pg, self.Tg, self.qtg, 0.0, 0.0)


            # Form a right hand side for integrating the hydrostatic equation to
            # determine the reference pressure
            ##_____________TO COMPILE______________
            def rhs(p, z):
                ret =  eos(t_to_entropy_c, eos_first_guess_entropy, np.exp(p),  self.qtg, self.sg)
                q_i = 0.0
                q_l = ret.ql
                T = ret.T
                return -g / (Rd * T * (1.0 - self.qtg + eps_vi * (self.qtg - q_l - q_i)))



            ##_____________TO COMPILE______________

            # Construct arrays for integration points
            z = np.array(Gr.z[Gr.gw - 1:-Gr.gw + 1])
            z_half = np.append([0.0], np.array(Gr.z_half[Gr.gw:-Gr.gw]))

            # We are integrating the log pressure so need to take the log of the
            # surface pressure
            p0 = np.log(self.Pg)

            p = np.zeros(Gr.nzg, dtype=np.double, order='c')
            p_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

            # Perform the integration
            p[Gr.gw - 1:-Gr.gw +1] = odeint(rhs, p0, z, hmax=1.0)[:, 0]
            p_half[Gr.gw:-Gr.gw] = odeint(rhs, p0, z_half, hmax=1.0)[1:, 0]

            # Set boundary conditions
            p[:Gr.gw - 1] = p[2 * Gr.gw - 2:Gr.gw - 1:-1]
            p[-Gr.gw + 1:] = p[-Gr.gw - 1:-2 * Gr.gw:-1]

            p_half[:Gr.gw] = p_half[2 * Gr.gw - 1:Gr.gw - 1:-1]
            p_half[-Gr.gw:] = p_half[-Gr.gw - 1:-2 * Gr.gw - 1:-1]

            p = np.exp(p)
            p_half = np.exp(p_half)

            p_ = p
            p_half_ = p_half

            # Compute reference state thermodynamic profiles
            #_____COMMENTED TO TEST COMPILATION_____________________
            for k in xrange(Gr.nzg):
                ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_[k], self.qtg, self.sg)
                temperature[k] = ret.T
                ql[k] = ret.ql
                qv[k] = self.qtg - (ql[k] + qi[k])
                alpha[k] = alpha_c(p_[k], temperature[k], self.qtg, qv[k])
                ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_half_[k], self.qtg, self.sg)
                temperature_half[k] = ret.T
                ql_half[k] = ret.ql
                qv_half[k] = self.qtg - (ql_half[k] + qi_half[k])
                alpha_half[k] = alpha_c(p_half_[k], temperature_half[k], self.qtg, qv_half[k])

            # Now do a sanity check to make sure that the Reference State entropy profile is uniform following
            # saturation adjustment
            for k in xrange(Gr.nzg):
                s = t_to_entropy_c(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])
                if np.abs(s - self.sg)/self.sg > 0.01:
                    print('Error in reference profiles entropy not constant !')
                    print('Likely error in saturation adjustment')





        # print(np.array(Gr.extract_local_ghosted(alpha_half,2)))
        self.alpha0_half = alpha_half
        self.alpha0 = alpha
        self.p0 = p_
        self.p0_half = p_half
        self.rho0 = 1.0 / np.array(self.alpha0)
        self.rho0_half = 1.0 / np.array(self.alpha0_half)

        Stats.add_reference_profile('alpha0')
        Stats.write_reference_profile('alpha0', alpha[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('alpha0_half')
        Stats.write_reference_profile('alpha0_half', alpha_half[Gr.gw:-Gr.gw])


        Stats.add_reference_profile('p0')
        Stats.write_reference_profile('p0', p_[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('p0_half')
        Stats.write_reference_profile('p0_half', p_half[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('rho0')
        Stats.write_reference_profile('rho0', 1.0 / np.array(alpha[Gr.gw:-Gr.gw]))
        Stats.add_reference_profile('rho0_half')
        Stats.write_reference_profile('rho0_half', 1.0 / np.array(alpha_half[Gr.gw:-Gr.gw]))

        # Stats.add_reference_profile('temperature0', Gr, Pa)
        # Stats.write_reference_profile('temperature0', temperature_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('ql0', Gr, Pa)
        # Stats.write_reference_profile('ql0', ql_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('qv0', Gr, Pa)
        # Stats.write_reference_profile('qv0', qv_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('qi0', Gr, Pa)
        # Stats.write_reference_profile('qi0', qi_half[Gr.dims.gw:-Gr.dims.gw], Pa)


        return

