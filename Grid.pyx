#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

#Adapated from PyCLES: https://github.com/pressel/pycles

cimport numpy as np
import numpy as np
import time
import netCDF4 as nc

cdef class Grid:
    '''
    A class for storing information about the LES grid.
    '''
    def __init__(self,namelist):
        '''

        :param namelist: Namelist dictionary
        :param Parallel: ParallelMPI class
        :return:
        '''

        #Get the grid spacing
        self.dz = namelist['grid']['dz']

        #Set the inverse grid spacing

        self.dzi = 1.0/self.dz

        #Get the grid dimensions and ghost points
        self.gw = namelist['grid']['gw']
        if namelist['meta']['casename'] == 'LES_driven_SCM':
            lesfolder = namelist['meta']['lesfolder']
            lesfile = namelist['meta']['lesfile']
            self.les_filename = lesfolder + 'Stats.' + lesfile +'.nc'
            les_data = nc.Dataset(self.les_filename,'r')
            z_top = np.max(les_data.groups['timeseries'].variables['cloud_top'])
            self.nz = int((z_top - z_top%self.dz)/self.dz*1.5)
        else:
            self.nz = namelist['grid']['nz']
        self.nzg = self.nz + 2 * self.gw

        self.z_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.z = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')


        cdef int i, count = 0
        for i in xrange(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1


        return




