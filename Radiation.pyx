#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
import netCDF4 as nc
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from NetCDFIO cimport  NetCDFIO_Stats
from TimeStepping cimport TimeStepping
from Variables cimport GridMeanVariables, VariablePrognostic
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin
import math as mt
from scipy.interpolate import interp2d

cdef class RadiationBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        self.dTdt = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.dqtdt = np.zeros((Gr.nzg,), dtype=np.double, order='c')

        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
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
    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
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
        cdef:
            Py_ssize_t tt, k, ind1, ind2

        RadiationBase.initialize(self, Gr, GMV, TS)
        self.dTdt = np.zeros(Gr.nzg, dtype=np.double)
        self.rad_time = np.linspace(10,360,36)*60
        z_in = np.array([42.5, 200.92, 456.28, 743, 1061.08, 1410.52, 1791.32, 2203.48, 2647,3121.88, 3628.12,
                         4165.72, 4734.68, 5335, 5966.68, 6629.72, 7324.12,
                         8049.88, 8807, 9595.48, 10415.32, 11266.52, 12149.08, 13063, 14008.28,
                         14984.92, 15992.92, 17032.28, 18103, 19205.08, 20338.52, 21503.32, 22699.48])
        rad_in   = np.array([[-1.386, -1.927, -2.089, -1.969, -1.805, -1.585, -1.406, -1.317, -1.188, -1.106, -1.103, -1.025,
                              -0.955, -1.045, -1.144, -1.119, -1.068, -1.092, -1.196, -1.253, -1.266, -1.306,  -0.95,  0.122,
                               0.255,  0.258,  0.322,  0.135,      0,      0,      0,      0,      0],
                             [ -1.23, -1.824, -2.011, -1.895, -1.729, -1.508, -1.331, -1.241, -1.109, -1.024, -1.018,  -0.94,
                              -0.867, -0.953, -1.046, -1.018, -0.972, -1.006, -1.119, -1.187, -1.209, -1.259, -0.919,  0.122,
                               0.264,  0.262,  0.326,  0.137,      0,      0,      0,      0,     0],
                             [-1.043, -1.692, -1.906, -1.796,  -1.63,  -1.41, -1.233, -1.142,  -1.01,  -0.92, -0.911, -0.829,
                              -0.754, -0.837, -0.923,  -0.89, -0.847, -0.895, -1.021, -1.101, -1.138, -1.201,  -0.88,  0.131,
                               0.286,  0.259,  0.332,   0.14,      0,      0,      0,      0,      0],
                             [-0.944, -1.613, -1.832,  -1.72, -1.555, -1.339, -1.163, -1.068, -0.935, -0.846, -0.835,  -0.75,
                              -0.673, -0.751, -0.833, -0.798,  -0.76, -0.817, -0.952, -1.042, -1.088, -1.159, -0.853,  0.138,
                               0.291,  0.265,  0.348,  0.136,      0,      0,      0,      0,      0],
                             [-0.833, -1.526, -1.757, -1.648, -1.485,  -1.27, -1.093, -0.998, -0.867, -0.778, -0.761, -0.672,
                              -0.594, -0.671, -0.748, -0.709, -0.676, -0.742, -0.887, -0.986, -1.041, -1.119, -0.825,  0.143,
                               0.296,  0.271,  0.351,  0.138,      0,      0,      0,      0,      0],
                             [-0.719, -1.425, -1.657,  -1.55, -1.392, -1.179, -1.003, -0.909, -0.778, -0.688, -0.667, -0.573,
                              -0.492, -0.566, -0.639, -0.596, -0.568, -0.647, -0.804, -0.914, -0.981,  -1.07, -0.793,  0.151,
                               0.303,  0.279,  0.355,  0.141,      0,      0,      0,      0,      0],
                             [-0.724, -1.374, -1.585, -1.482, -1.328, -1.116, -0.936, -0.842, -0.715, -0.624, -0.598, -0.503,
                              -0.421, -0.494, -0.561, -0.514,  -0.49,  -0.58, -0.745, -0.863, -0.938, -1.035, -0.764,  0.171,
                               0.291,  0.284,  0.358,  0.144,      0,      0,      0,      0,      0],
                             [-0.587,  -1.28, -1.513, -1.416, -1.264, -1.052, -0.874, -0.781, -0.655, -0.561, -0.532, -0.436,
                              -0.354, -0.424, -0.485, -0.435, -0.417, -0.517, -0.691, -0.817, -0.898,     -1,  -0.74,  0.176,
                               0.297,  0.289,   0.36,  0.146,      0,      0,      0,      0,      0],
                             [-0.506, -1.194, -1.426, -1.332, -1.182, -0.972, -0.795, -0.704, -0.578,  -0.48, -0.445, -0.347,
                              -0.267, -0.336, -0.391, -0.337, -0.325, -0.436,  -0.62, -0.756, -0.847,  -0.96, -0.714,   0.18,
                               0.305,  0.317,  0.348,  0.158,      0,      0,      0,      0,      0],
                             [-0.472,  -1.14, -1.364, -1.271, -1.123, -0.914, -0.738, -0.649, -0.522, -0.422, -0.386, -0.287,
                              -0.207, -0.273, -0.322, -0.267,  -0.26, -0.379, -0.569, -0.712, -0.811, -0.931, -0.696,  0.183,
                               0.311,   0.32,  0.351,   0.16,      0,      0,      0,      0,     0],
                             [-0.448, -1.091, -1.305, -1.214, -1.068, -0.858, -0.682, -0.594, -0.469, -0.368, -0.329, -0.229,
                              -0.149, -0.213, -0.257,   -0.2, -0.199, -0.327, -0.523, -0.668, -0.774, -0.903, -0.678,  0.186,
                               0.315,  0.323,  0.355,  0.162,      0,      0,      0,      0,      0],
                             [-0.405, -1.025, -1.228, -1.139, -0.996, -0.789, -0.615, -0.527, -0.402,   -0.3, -0.256, -0.156,
                              -0.077, -0.136, -0.173, -0.115, -0.121, -0.259, -0.463, -0.617, -0.732, -0.869, -0.656,   0.19,
                               0.322,  0.326,  0.359,  0.164,      0,      0,      0,      0,      0],
                             [-0.391, -0.983, -1.174, -1.085, -0.945, -0.739, -0.566, -0.478, -0.354, -0.251, -0.205, -0.105,
                              -0.027, -0.082, -0.114, -0.056, -0.069, -0.213,  -0.42, -0.579, -0.699,  -0.84, -0.642,  0.173,
                               0.327,  0.329,  0.362,  0.165,      0,      0,      0,      0,      0],
                             [-0.385, -0.946, -1.121, -1.032, -0.898, -0.695, -0.523, -0.434, -0.307, -0.203, -0.157, -0.057,
                               0.021, -0.031, -0.059, -0.001, -0.018, -0.168, -0.381, -0.546, -0.672, -0.819, -0.629,  0.176,
                               0.332,  0.332,  0.364,  0.166,      0,      0,      0,      0,      0],
                             [-0.383, -0.904, -1.063, -0.972, -0.834, -0.632, -0.464, -0.378, -0.251, -0.144, -0.096,  0.001,
                               0.079,  0.032,  0.011,  0.069,  0.044, -0.113, -0.332, -0.504, -0.637, -0.791, -0.611,  0.181,
                               0.338,  0.335,  0.367,  0.167,      0,      0,      0,      0,      0],
                             [-0.391, -0.873, -1.016, -0.929, -0.794, -0.591, -0.423, -0.337, -0.212, -0.104, -0.056,  0.043,
                               0.121,  0.077,  0.058,  0.117,  0.088, -0.075, -0.298, -0.475, -0.613, -0.772, -0.599,  0.183,
                               0.342,  0.337,   0.37,  0.168,      0,      0,      0,      0,      0],
                             [-0.359, -0.836, -0.976, -0.888, -0.755, -0.554, -0.386,   -0.3, -0.175, -0.067, -0.018,  0.081,
                                0.16,  0.119,  0.103,  0.161,  0.129, -0.039, -0.266, -0.448, -0.591, -0.755, -0.587,  0.187,
                               0.345,  0.339,  0.372,  0.169,      0,      0,      0,      0,     0],
                             [-0.328, -0.792, -0.928, -0.842, -0.709, -0.508, -0.341, -0.256, -0.131, -0.022,  0.029,  0.128,
                               0.208,   0.17,  0.158,  0.216,  0.179,  0.005, -0.228, -0.415, -0.564, -0.733, -0.573,   0.19,
                               0.384,  0.313,  0.375,   0.17,      0,      0,      0,      0,      0],
                             [-0.324, -0.767, -0.893, -0.807, -0.676, -0.476,  -0.31, -0.225, -0.101,  0.008,   0.06,  0.159,
                               0.239,  0.204,  0.195,  0.252,  0.212,  0.034, -0.203, -0.394, -0.546, -0.719, -0.564,  0.192,
                               0.386,  0.315,  0.377,  0.171,      0,      0,      0,      0,      0],
                             [ -0.31,  -0.74,  -0.86, -0.775, -0.647, -0.449, -0.283, -0.197, -0.073,  0.036,  0.089,  0.188,
                               0.269,  0.235,  0.229,  0.285,  0.242,  0.061, -0.179, -0.374,  -0.53, -0.706, -0.556,  0.194,
                               0.388,  0.317,  0.402,  0.158,      0,      0,      0,      0,      0],
                             [-0.244, -0.694, -0.818,  -0.73, -0.605, -0.415, -0.252, -0.163, -0.037,  0.072,  0.122,   0.22,
                               0.303,  0.273,  0.269,  0.324,  0.277,  0.093, -0.152,  -0.35,  -0.51, -0.691, -0.546,  0.196,
                               0.39,   0.32,  0.403,  0.159,      0,      0,      0,      0,      0],
                             [-0.284, -0.701, -0.803, -0.701, -0.568, -0.381, -0.225, -0.142, -0.017,  0.092,  0.143,  0.242,
                               0.325,  0.298,  0.295,   0.35,    0.3,  0.112, -0.134, -0.334, -0.497,  -0.68,  -0.54,  0.198,
                               0.392,  0.321,  0.404,   0.16,      0,      0,      0,      0,      0],
                             [-0.281, -0.686, -0.783,  -0.68, -0.547, -0.359, -0.202, -0.119,  0.005,  0.112,  0.163,  0.261,
                               0.345,  0.321,  0.319,  0.371,  0.319,   0.13, -0.118, -0.321, -0.486, -0.671, -0.534,  0.199,
                               0.393,  0.323,  0.405,  0.161,      0,      0,      0,      0,      0],
                             [-0.269, -0.667,  -0.76, -0.655, -0.522, -0.336, -0.181, -0.096,  0.029,  0.136,  0.188,  0.286,
                                0.37,  0.346,  0.345,  0.396,  0.342,   0.15, -0.102, -0.307, -0.473, -0.661, -0.528,    0.2,
                               0.393,  0.324,  0.405,  0.162,      0,      0,      0,      0,      0],
                             [-0.255, -0.653, -0.747, -0.643, -0.511, -0.325, -0.169, -0.082,  0.042,  0.149,  0.204,  0.304,
                               0.388,  0.363,  0.36 ,  0.409,  0.354,  0.164, -0.085, -0.289, -0.457, -0.649, -0.523,  0.193,
                               0.394,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.265,  -0.65, -0.739, -0.634,   -0.5, -0.314, -0.159, -0.072,  0.052,  0.159,  0.215,  0.316,
                               0.398,  0.374,  0.374,  0.424,   0.37,  0.181, -0.065, -0.265, -0.429, -0.627, -0.519,   0.18,
                               0.394,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.276, -0.647, -0.731, -0.626, -0.492, -0.307, -0.152, -0.064,  0.058,  0.166,  0.227,  0.329,
                               0.411,  0.389,   0.39,  0.441,  0.389,  0.207, -0.032, -0.228, -0.394, -0.596, -0.494,  0.194,
                               0.376,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.271, -0.646,  -0.73, -0.625, -0.489, -0.303, -0.149, -0.061,  0.062,  0.169,  0.229,  0.332,
                               0.412,  0.388,  0.389,  0.439,  0.387,  0.206, -0.028, -0.209, -0.347, -0.524, -0.435,  0.195,
                               0.381,  0.313,  0.405,  0.162,      0,      0,      0,      0,      0],
                             [-0.267, -0.647, -0.734, -0.628,  -0.49, -0.304, -0.151, -0.062,  0.061,  0.168,  0.229,  0.329,
                               0.408,  0.385,  0.388,  0.438,  0.386,  0.206, -0.024, -0.194, -0.319,  -0.48,  -0.36,  0.318,
                               0.405,  0.335,  0.394,  0.162,      0,      0,      0,      0,      0],
                             [-0.274, -0.656, -0.745,  -0.64,   -0.5, -0.313, -0.158, -0.068,  0.054,  0.161,  0.223,  0.325,
                               0.402,  0.379,  0.384,  0.438,  0.392,  0.221,  0.001, -0.164, -0.278, -0.415, -0.264,  0.445,
                               0.402,  0.304,  0.389,  0.157,      0,      0,      0,      0,      0],
                             [-0.289, -0.666, -0.753, -0.648, -0.508,  -0.32, -0.164, -0.073,  0.049,  0.156,   0.22,  0.321,
                               0.397,  0.374,  0.377,   0.43,  0.387,  0.224,  0.014, -0.139, -0.236, -0.359, -0.211,  0.475,
                                 0.4,  0.308,  0.375,  0.155,      0,      0,      0,      0,      0],
                             [-0.302, -0.678, -0.765, -0.659, -0.517, -0.329, -0.176, -0.085,  0.038,  0.145,  0.208,   0.31,
                               0.386,  0.362,  0.366,  0.421,  0.381,  0.224,  0.022, -0.119, -0.201,   -0.3, -0.129,  0.572,
                               0.419,  0.265,  0.364,  0.154,      0,      0,      0,      0,      0],
                             [-0.314, -0.696, -0.786, -0.681, -0.539, -0.349, -0.196, -0.105,  0.019,  0.127,  0.189,  0.289,
                               0.364,   0.34,  0.346,  0.403,   0.37,  0.222,  0.036, -0.081, -0.133, -0.205, -0.021,  0.674,
                               0.383,  0.237,  0.359,  0.151,      0,      0,      0,      0,      0],
                             [-0.341, -0.719, -0.807, -0.702, -0.558, -0.367, -0.211,  -0.12,  0.003,  0.111,  0.175,  0.277,
                               0.351,  0.325,  0.331,   0.39,   0.36,  0.221,  0.048, -0.046, -0.074, -0.139,  0.038,  0.726,
                               0.429,  0.215,  0.347,  0.151,      0,      0,      0,      0,      0],
                             [ -0.35, -0.737, -0.829, -0.724, -0.577, -0.385, -0.229, -0.136, -0.011,  0.098,  0.163,  0.266,
                               0.338,   0.31,  0.316,  0.378,  0.354,  0.221,  0.062, -0.009, -0.012, -0.063,  0.119,  0.811,
                               0.319,  0.201,  0.343,  0.148,      0,      0,      0,      0,      0],
                             [-0.344,  -0.75, -0.856, -0.757, -0.607, -0.409,  -0.25, -0.156, -0.033,  0.076,  0.143,  0.246,
                               0.316,  0.287,  0.293,  0.361,  0.345,  0.225,  0.082,  0.035,  0.071,  0.046,  0.172,  0.708,
                               0.255,   0.21,  0.325,  0.146,      0,      0,      0,      0,      0]])/86400

        A = np.interp(Gr.z_half,z_in,rad_in[0,:])
        for tt in xrange(1,36):
            A = np.vstack((A, np.interp(Gr.z_half,z_in,rad_in[tt,:])))
        self.rad = A # store matrix in self
        ind1 = int(mt.trunc(10.0/600.0))
        ind2 = int(mt.ceil(10.0/600.0))
        for k in xrange(Gr.nzg):
            if 10%600.0 == 0:
                self.dTdt[k] = self.rad[ind1,k]
            else:
                self.dTdt[k]    = (self.rad[ind2,k]-self.rad[ind1,k])/\
                                  (self.rad_time[ind2]-self.rad_time[ind1])*(10.0)+self.rad[ind1,k]
        return

    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k, ind1, ind2
            double qv

        ind2 = int(mt.ceil(TS.t/600.0))
        ind1 = int(mt.trunc(TS.t/600.0))
        for k in xrange(Gr.nzg):
            if Gr.z_half[k] >= 22699.48:
                self.dTdt[k] = 0.0
            else:
                if TS.t<600.0: # first 10 min use the radiative forcing of t=10min (as in the paper)
                    self.dTdt[k] = self.rad[0,k]
                elif TS.t<21600.0 and ind2<36:
                    if TS.t%600.0 == 0:
                        self.dTdt[k] = self.rad[ind1,k]
                    else:
                        self.dTdt[k] = (self.rad[ind2,k]-self.rad[ind1,k])\
                                                 /(self.rad_time[ind2]-self.rad_time[ind1])\
                                                 *(TS.t-self.rad_time[ind1])+self.rad[ind1,k]
                else:
                    self.dTdt[k] = self.rad[35,k]


        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(Ref.p0_half[k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
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

        self.f_rad = np.zeros((Gr.nzg + 1), dtype=np.double, order='c') # radiative flux at cell edges
        return

        """
        see eq. 3 in Stevens et. al. 2005 DYCOMS paper
        """

    cpdef calculate_radiation(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double zi
            double rhoi

        # find zi (level of 8.0 g/kg isoline of qt)
        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            if (GMV.QT.values[k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                zi     = Gr.z[idx_zi]
                rhoi   = Ref.rho0[idx_zi]
                break

        # cloud-top cooling
        q_0 = 0.0

        self.f_rad = np.zeros((Gr.nzg + 1), dtype=np.double, order='c')
        self.f_rad[Gr.nzg] = self.F0 * np.exp(-q_0)
        for k in xrange(Gr.nzg - 1, -1, -1):
            q_0           += self.kappa * Ref.rho0_half[k] * GMV.QL.values[k] * Gr.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[0] += self.F1 * np.exp(-q_1)
        for k in xrange(1, Gr.nzg + 1):
            q_1           += self.kappa * Ref.rho0_half[k - 1] * GMV.QL.values[k - 1] * Gr.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in xrange(0, Gr.nzg):
            if Gr.z[k] > zi:
                cbrt_z         = cbrt(Gr.z[k] - zi)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)
        # condition at the top
        cbrt_z                   = cbrt(Gr.z[k] + Gr.dz - zi)
        self.f_rad[Gr.nzg] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)

        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / Gr.dz / Ref.rho0_half[k] / dycoms_cp
        return

    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k
            double qv

        self.calculate_radiation(Ref, Gr, GMV, TS)
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Apply radiative temperature tendency
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k]  += self.convert_forcing_prog_fp(Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('rad_dTdt', self.dTdt[self.Gr.gw     : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('rad_flux', self.f_rad[self.Gr.gw + 1 : self.Gr.nzg - self.Gr.gw + 1])
        return


cdef class RadiationLES(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
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

    cpdef update(self, ReferenceState Ref, Grid Gr, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t i, k

        i = int(TS.t/TS.dt)
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.radiation[k] = self.convert_forcing_prog_fp(Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], qv, self.dtdt_rad[i,k])
            GMV.H.tendencies[k] += GMV.H.radiation[k]
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        return