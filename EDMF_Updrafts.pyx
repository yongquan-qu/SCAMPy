#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
from thermodynamic_functions cimport  *
from microphysics_functions cimport  *
import cython
cimport Grid
cimport ReferenceState
cimport EDMF_Rain
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from libc.math cimport fmax, fmin


cdef class UpdraftVariable:
    def __init__(self, nu, nz, loc, kind, name, units):
        self.values = np.zeros((nu,nz),dtype=np.double, order='c')
        self.old = np.zeros((nu,nz),dtype=np.double, order='c')  # needed for prognostic updrafts
        self.new = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.tendencies = np.zeros((nu,nz),dtype=np.double, order='c')
        self.flux = np.zeros((nu,nz),dtype=np.double, order='c')
        self.bulkvalues = np.zeros((nz,), dtype=np.double, order = 'c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid.Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        n_updrafts = np.shape(self.values)[0]

        if self.name == 'w':
            for i in xrange(n_updrafts):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in xrange(1,Gr.gw):
                    self.values[i,start_high+ k] = -self.values[i,start_high - k ]
                    self.values[i,start_low- k] = -self.values[i,start_low + k  ]
        else:
            for k in xrange(Gr.gw):
                for i in xrange(n_updrafts):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]

        return

cdef class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Grid.Grid Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t i, k

        self.W    = UpdraftVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )

        self.Area = UpdraftVariable(nu, nzg, 'half', 'scalar', 'area_fraction','[-]' )
        self.QT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.RH = UpdraftVariable(nu, nzg, 'half', 'scalar', 'RH','%' )

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )

        self.THL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )

        if namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            try:
                use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
            except:
                use_steady_updrafts = False
            if use_steady_updrafts:
                self.prognostic = False
            else:
                self.prognostic = True
            self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        else:
            self.prognostic = False
            self.updraft_fraction = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']

        # cloud and rain diagnostics for output
        self.cloud_fraction = np.zeros((nzg,), dtype=np.double, order='c')

        self.cloud_base     = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_top      = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_cover    = np.zeros((nu,),  dtype=np.double, order='c')
        self.updraft_top    = np.zeros((nu,),  dtype=np.double, order='c')

        self.lwp = 0.
        return

    cpdef initialize(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    if self.prognostic:
                        self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                    else:
                        self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = GMV.QT.values[k]
                    self.QL.values[i,k] = GMV.QL.values[k]
                    self.H.values[i,k]  = GMV.H.values[k]
                    self.T.values[i,k]  = GMV.T.values[k]
                    self.B.values[i,k]  = 0.0

                self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    cpdef initialize_SaturatedBubble(self, GridMeanVariables GMV, ReferenceState.ReferenceState Ref):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        # using LES diagnostics at simul time 10sec as input; updraft identification varies in critera
        # w>0.01 or b>1e-4 or (w>0.01 & b>1e-4)

        # # criterion 1: w>0.01
        # # w only is not a rational criterion and thus deleted from here

        # criterion 2: b>1e-4
        z_in = np.array([
                          75.,  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,
                         525.,  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,
                         975., 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375.,
                        1425., 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825.,
                        1875., 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275.,
                        2325., 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725.,
                        2775., 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175.,
                        3225., 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625.,
                        3675., 3725., 3775., 3825., 3875., 3925.
        ])

        thetal_in = np.array([
                        295.6455, 295.6453, 295.6462, 295.6529, 295.6603, 295.6746,
                        295.6911, 295.7059, 295.727 , 295.7569, 295.7825, 295.8095,
                        295.8373, 295.8774, 295.907 , 295.9505, 295.9799, 296.0255,
                        296.0534, 296.0989, 296.1244, 296.1683, 296.212 , 296.2313,
                        296.271 , 296.3096, 296.3463, 296.3525, 296.3831, 296.4112,
                        296.4356, 296.4567, 296.4745, 296.4567, 296.4662, 296.4706,
                        296.4719, 296.4681, 296.4598, 296.447 , 296.4298, 296.4081,
                        296.382 , 296.3516, 296.3169, 296.3092, 296.266 , 296.2191,
                        296.1686, 296.1149, 296.058 , 296.0253, 295.9624, 295.8974,
                        295.8305, 295.785 , 295.7146, 295.6433, 295.5914, 295.5185,
                        295.4635, 295.3908, 295.3193, 295.2631, 295.2058, 295.1383,
                        295.0826, 295.0281, 294.9671, 294.9159, 294.8673, 294.82  ,
                        294.7803, 294.7383, 294.7019, 294.6682, 294.6376, 294.6108
        ])
        Area_in = np.array([
                        0.025, 0.05 , 0.065, 0.075, 0.085, 0.09 , 0.095, 0.105, 0.11 ,
                        0.11 , 0.115, 0.12 , 0.125, 0.125, 0.13 , 0.13 , 0.135, 0.135,
                        0.14 , 0.14 , 0.145, 0.145, 0.145, 0.15 , 0.15 , 0.15 , 0.15 ,
                        0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.16 , 0.16 , 0.16 ,
                        0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 ,
                        0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.15 , 0.15 , 0.15 ,
                        0.15 , 0.145, 0.145, 0.145, 0.14 , 0.14 , 0.135, 0.135, 0.135,
                        0.13 , 0.125, 0.125, 0.12 , 0.115, 0.115, 0.11 , 0.105, 0.1  ,
                        0.09 , 0.085, 0.075, 0.065, 0.05 , 0.03
        ])

        W_in = np.array([
                        0.0072, 0.0111, 0.0144, 0.0175, 0.0201, 0.0231, 0.0259, 0.0273,
                        0.0294, 0.0328, 0.0347, 0.0363, 0.0376, 0.0407, 0.0417, 0.0447,
                        0.0453, 0.0481, 0.0483, 0.051 , 0.0508, 0.0532, 0.0556, 0.0549,
                        0.057 , 0.0592, 0.0612, 0.0597, 0.0615, 0.0632, 0.0647, 0.0662,
                        0.0675, 0.065 , 0.066 , 0.0669, 0.0677, 0.0684, 0.0689, 0.0693,
                        0.0696, 0.0698, 0.0698, 0.0697, 0.0694, 0.0729, 0.0724, 0.0718,
                        0.071 , 0.0701, 0.0692, 0.0717, 0.0705, 0.0691, 0.0677, 0.0696,
                        0.0679, 0.0662, 0.0675, 0.0656, 0.0666, 0.0645, 0.0623, 0.0629,
                        0.0632, 0.0608, 0.0608, 0.0606, 0.0582, 0.0577, 0.0571, 0.0564,
                        0.057 , 0.0558, 0.0558, 0.0554, 0.0554, 0.0551
        ])

        QL_in = np.array([
                        0.0008, 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
                        0.0017, 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023, 0.0024,
                        0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031, 0.0032,
                        0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.004 ,
                        0.0041, 0.0042, 0.0043, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049,
                        0.005 , 0.0051, 0.0052, 0.0053, 0.0055, 0.0056, 0.0057, 0.0058,
                        0.0059, 0.006 , 0.0061, 0.0063, 0.0064, 0.0065, 0.0066, 0.0067,
                        0.0068, 0.007 , 0.0071, 0.0072, 0.0073, 0.0074, 0.0076, 0.0077,
                        0.0078, 0.0079, 0.008 , 0.0081, 0.0082, 0.0083, 0.0085, 0.0086,
                        0.0087, 0.0088, 0.0089, 0.009 , 0.0091, 0.0092
        ])

        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.nzg):
                if self.Gr.z_half[k+self.Gr.gw]<=z_in.max():
                    self.W.values[i,k] = np.interp(self.Gr.z_half[k+self.Gr.gw],z_in,W_in)
                    # self.W.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    self.Area.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,Area_in) #self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = 0.0196
                    self.H.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,thetal_in)
                    self.QL.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,QL_in)

                    sa = eos(
                        t_to_thetali_c,
                        eos_first_guess_thetal,
                        Ref.p0_half[k],
                        self.QT.values[i,k],
                        self.H.values[i,k]
                    )
                    self.QL.values[i,k] = sa.ql
                    self.T.values[i,k] = sa.T

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)
        self.W.set_bcs(self.Gr)

        self.set_means(GMV)

        return


    cpdef initialize_DryBubble(self, GridMeanVariables GMV, ReferenceState.ReferenceState Ref):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        # criterion 2: b>1e-4
        z_in = np.array([
                          75.,  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,
                         525.,  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,
                         975., 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375.,
                        1425., 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825.,
                        1875., 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275.,
                        2325., 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725.,
                        2775., 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175.,
                        3225., 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625.,
                        3675., 3725., 3775., 3825., 3875., 3925.
        ])

        thetal_in = np.array([
                        299.9882, 299.996 , 300.0063, 300.0205, 300.04  , 300.0594,
                        300.0848, 300.1131, 300.1438, 300.1766, 300.2198, 300.2567,
                        300.2946, 300.3452, 300.3849, 300.4245, 300.4791, 300.5182,
                        300.574 , 300.6305, 300.6668, 300.7222, 300.7771, 300.8074,
                        300.8591, 300.9092, 300.9574, 300.9758, 301.0182, 301.0579,
                        301.0944, 301.1276, 301.1572, 301.1515, 301.1729, 301.1902,
                        301.2033, 301.2122, 301.2167, 301.2169, 301.2127, 301.2041,
                        301.1913, 301.1743, 301.1533, 301.1593, 301.1299, 301.097 ,
                        301.0606, 301.0212, 300.9788, 300.9607, 300.9125, 300.8625,
                        300.8108, 300.7806, 300.7256, 300.6701, 300.6338, 300.5772,
                        300.5212, 300.482 , 300.4272, 300.3875, 300.3354, 300.2968,
                        300.2587, 300.2216, 300.1782, 300.1452, 300.1143, 300.0859,
                        300.0603, 300.0408, 300.0211, 300.0067, 299.9963, 299.9884
        ])

        Area_in = np.array([
                        0.04 , 0.055, 0.07 , 0.08 , 0.085, 0.095, 0.1  , 0.105, 0.11 ,
                        0.115, 0.115, 0.12 , 0.125, 0.125, 0.13 , 0.135, 0.135, 0.14 ,
                        0.14 , 0.14 , 0.145, 0.145, 0.145, 0.15 , 0.15 , 0.15 , 0.15 ,
                        0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.16 , 0.16 , 0.16 ,
                        0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 ,
                        0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.15 , 0.15 , 0.15 ,
                        0.15 , 0.145, 0.145, 0.145, 0.14 , 0.14 , 0.14 , 0.135, 0.135,
                        0.13 , 0.13 , 0.125, 0.12 , 0.115, 0.115, 0.11 , 0.105, 0.1  ,
                        0.095, 0.085, 0.08 , 0.07 , 0.055, 0.04
        ])

        W_in = np.array([
                        0.017 , 0.0266, 0.0344, 0.0417, 0.0495, 0.0546, 0.061 , 0.0668,
                        0.0721, 0.0768, 0.0849, 0.0887, 0.092 , 0.0996, 0.1019, 0.1037,
                        0.1106, 0.1114, 0.1179, 0.1243, 0.1238, 0.1297, 0.1355, 0.1335,
                        0.1387, 0.1437, 0.1485, 0.1448, 0.1489, 0.1527, 0.1564, 0.1597,
                        0.1628, 0.1565, 0.1588, 0.1609, 0.1626, 0.1641, 0.1652, 0.166 ,
                        0.1665, 0.1667, 0.1666, 0.1662, 0.1655, 0.1736, 0.1722, 0.1706,
                        0.1686, 0.1664, 0.1639, 0.1698, 0.1667, 0.1634, 0.1599, 0.1641,
                        0.1601, 0.1559, 0.1589, 0.1543, 0.1496, 0.1514, 0.1464, 0.1475,
                        0.1422, 0.1425, 0.1424, 0.1419, 0.1361, 0.135 , 0.1335, 0.1316,
                        0.1294, 0.1302, 0.1271, 0.1264, 0.1269, 0.1256
        ])

        QT_in = np.array([
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ])

        QL_in = np.array([
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ])

        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.nzg):
                if self.Gr.z_half[k+self.Gr.gw]<=z_in.max():
                    self.W.values[i,k] = np.interp(self.Gr.z_half[k+self.Gr.gw],z_in,W_in)
                    # self.W.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    self.Area.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,Area_in) #self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = 1.0e-5
                    self.H.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,thetal_in)
                    self.QL.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,QL_in)

                    sa = eos(
                        t_to_thetali_c,
                        eos_first_guess_thetal,
                        Ref.p0_half[k],
                        self.QT.values[i,k],
                        self.H.values[i,k]
                    )
                    self.QL.values[i,k] = sa.ql
                    self.T.values[i,k] = sa.T

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)
        self.W.set_bcs(self.Gr)

        self.set_means(GMV)

        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_RH')

        if self.H.name == 'thetal':
            Stats.add_profile('updraft_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_s')

        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_profile('updraft_cloud_fraction')

        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        Stats.add_ts('updraft_lwp')
        return

    cpdef set_means(self, GridMeanVariables GMV):

        cdef:
            Py_ssize_t i, k

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0
        self.RH.bulkvalues[:] = 0.0

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.Area.bulkvalues[k] > 1.0e-20:
                    for i in xrange(self.n_updrafts):
                        self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                        self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                        self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                        self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                        self.RH.bulkvalues[k] += self.Area.values[i,k] * self.RH.values[i,k]/self.Area.bulkvalues[k]
                        self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                        self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                            /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))

                else:
                    self.QT.bulkvalues[k] = GMV.QT.values[k]
                    self.QL.bulkvalues[k] = 0.0
                    self.H.bulkvalues[k] = GMV.H.values[k]
                    self.RH.bulkvalues[k] = GMV.RH.values[k]
                    self.T.bulkvalues[k] = GMV.T.values[k]
                    self.B.bulkvalues[k] = 0.0
                    self.W.bulkvalues[k] = 0.0

                if self.QL.bulkvalues[k] > 1e-8 and self.Area.bulkvalues[k] > 1e-3:
                    self.cloud_fraction[k] = 1.0
                else:
                    self.cloud_fraction[k] = 0.
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_new_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.new[i,k] = self.W.values[i,k]
                    self.Area.new[i,k] = self.Area.values[i,k]
                    self.QT.new[i,k] = self.QT.values[i,k]
                    self.QL.new[i,k] = self.QL.values[i,k]
                    self.H.new[i,k] = self.H.values[i,k]
                    self.THL.new[i,k] = self.THL.values[i,k]
                    self.T.new[i,k] = self.T.values[i,k]
                    self.B.new[i,k] = self.B.values[i,k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_old_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.old[i,k] = self.W.values[i,k]
                    self.Area.old[i,k] = self.Area.values[i,k]
                    self.QT.old[i,k] = self.QT.values[i,k]
                    self.QL.old[i,k] = self.QL.values[i,k]
                    self.H.old[i,k] = self.H.values[i,k]
                    self.THL.old[i,k] = self.THL.values[i,k]
                    self.T.old[i,k] = self.T.values[i,k]
                    self.B.old[i,k] = self.B.values[i,k]
        return

    # quick utility to set "tmp" arrays with values in the "new" arrays
    cpdef set_values_with_new(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.values[i,k] = self.W.new[i,k]
                    self.Area.values[i,k] = self.Area.new[i,k]
                    self.QT.values[i,k] = self.QT.new[i,k]
                    self.QL.values[i,k] = self.QL.new[i,k]
                    self.H.values[i,k] = self.H.new[i,k]
                    self.THL.values[i,k] = self.THL.new[i,k]
                    self.T.values[i,k] = self.T.new[i,k]
                    self.B.values[i,k] = self.B.new[i,k]
        return

    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState.ReferenceState Ref):

        Stats.write_profile('updraft_area', self.Area.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_w', self.W.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_qt', self.QT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_ql', self.QL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_RH', self.RH.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        if self.H.name == 'thetal':
            Stats.write_profile('updraft_thetal', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('updraft_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            #Stats.write_profile('updraft_thetal', self.THL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('updraft_temperature', self.T.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_buoyancy', self.B.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        self.upd_cloud_diagnostics(Ref)
        Stats.write_profile('updraft_cloud_fraction', self.cloud_fraction[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
        # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
        # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
        # at all. Thus total updraft cover is the sum of each updraft's cover
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base',  np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top',   np.amax(self.cloud_top))
        Stats.write_ts('updraft_lwp',         self.lwp)
        return

    cpdef upd_cloud_diagnostics(self, ReferenceState.ReferenceState Ref):
        cdef Py_ssize_t i, k
        self.lwp = 0.

        for i in xrange(self.n_updrafts):
            #TODO check the setting of ghost point z_half

            self.cloud_base[i] = self.Gr.z_half[self.Gr.nzg-self.Gr.gw-1]
            self.cloud_top[i] = 0.0
            self.updraft_top[i] = 0.0
            self.cloud_cover[i] = 0.0

            for k in xrange(self.Gr.gw,self.Gr.nzg-self.Gr.gw):

                if self.Area.values[i,k] > 1e-3:
                    self.updraft_top[i] = fmax(self.updraft_top[i], self.Gr.z_half[k])
                    self.lwp += Ref.rho0_half[k] * self.QL.values[i,k] * self.Area.values[i,k] * self.Gr.dz

                    if self.QL.values[i,k] > 1e-8:
                        self.cloud_base[i]  = fmin(self.cloud_base[i],  self.Gr.z_half[k])
                        self.cloud_top[i]   = fmax(self.cloud_top[i],   self.Gr.z_half[k])
                        self.cloud_cover[i] = fmax(self.cloud_cover[i], self.Area.values[i,k])

        return


cdef class UpdraftThermodynamics:
    def __init__(self, n_updraft, Grid.Grid Gr,
                 ReferenceState.ReferenceState Ref, UpdraftVariables UpdVar,
                 RainVariables Rain):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft

        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        # rain source from each updraft from all sub-timesteps
        self.prec_source_h  = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')

        # rain source from all updrafts from all sub-timesteps
        self.prec_source_h_tot  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.prec_source_qt_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')

        return

    cpdef clear_precip_sources(self):
        """
        clear precipitation source terms for QT and H from each updraft
        """
        self.prec_source_qt[:,:] = 0.
        self.prec_source_h[:,:]  = 0.
        return

    cpdef update_total_precip_sources(self):
        """
        sum precipitation source terms for QT and H from all sub-timesteps
        """
        self.prec_source_h_tot  = np.sum(self.prec_source_h,  axis=0)
        self.prec_source_qt_tot = np.sum(self.prec_source_qt, axis=0)
        return

    cpdef buoyancy(self, UpdraftVariables UpdVar, EnvironmentVariables EnvVar,
                   GridMeanVariables GMV, bint extrap):
        cdef:
            Py_ssize_t k, i
            double alpha, qv, qt, t, h
            Py_ssize_t gw = self.Gr.gw

        # print '-- upd thermo buoy --'

        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)

        if not extrap:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.nzg):
                        if UpdVar.Area.values[i,k] > 0.0:
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
                        else:
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                        UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], UpdVar.QT.values[i,k],
                                                    UpdVar.QL.values[i,k], 0.0, UpdVar.T.values[i,k])
        else:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        # if k < 5:
                        #     with gil:
                        #         print 'k: ' + str(k)
                        #         print 'area k: ' + str( UpdVar.Area.values[i,k] )
                        #         print 'area k-1: ' + str( UpdVar.Area.values[i,k-1] )
                        if UpdVar.Area.values[i,k] > 0.0:

                            qt = UpdVar.QT.values[i,k]
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            h = UpdVar.H.values[i,k]
                            t = UpdVar.T.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)

                            # if k < 5:
                            #     with gil:
                            #         print 'k: ' + str(k)
                            #         print 'if choice'
                            #         print 'b: ' + str( UpdVar.B.values[i,k] )

                        elif UpdVar.Area.values[i,k-1] > 0.0 and k>self.Gr.gw:
                            # if k < 5:
                            #     with gil:
                            #         print 'k: ' + str(k)
                            #         print 'elif choice'
                            sa = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                                     qt, h)
                            qt -= sa.ql
                            qv = qt
                            t = sa.T
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)
                        else:
                            # if k < 5:
                            #     with gil:
                            #         print 'k: ' + str(k)
                            #         print 'b upd <- env'
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                            UpdVar.RH.values[i,k] = EnvVar.RH.values[k]


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
                for i in xrange(self.n_updraft):
                    GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
                for i in xrange(self.n_updraft):
                    UpdVar.B.values[i,k] -= GMV.B.values[k]
                EnvVar.B.values[k] -= GMV.B.values[k]

        return

    cpdef microphysics(self, UpdraftVariables UpdVar, RainVariables Rain, double dt):
        """
        compute precipitation source terms
        """
        cdef:
            Py_ssize_t k, i

            rain_struct rst
            mph_struct  mph
            eos_struct  sa

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):

                    # autoconversion and accretion
                    mph = microphysics_rain_src(
                        Rain.rain_model,
                        UpdVar.QT.new[i,k],
                        UpdVar.QL.new[i,k],
                        Rain.Upd_QR.values[k],
                        UpdVar.Area.new[i,k],
                        UpdVar.T.new[i,k],
                        self.Ref.p0_half[k],
                        self.Ref.rho0_half[k],
                        dt
                    )

                    # update Updraft.new
                    UpdVar.QT.new[i,k] = mph.qt
                    UpdVar.QL.new[i,k] = mph.ql
                    UpdVar.H.new[i,k]  = mph.thl

                    # update rain sources of state variables
                    self.prec_source_qt[i,k] -= mph.qr_src * UpdVar.Area.new[i,k]
                    self.prec_source_h[i,k]  += mph.thl_rain_src * UpdVar.Area.new[i,k]
        return
