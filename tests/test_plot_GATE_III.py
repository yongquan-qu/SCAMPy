import sys
sys.path.insert(0, "./")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import pprint as pp
import numpy as np

import main as scampy
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('GATE_III')
    # chenge the defaults  
    #setup["namelist"]['stats_io']['frequency'] = setup["namelist"]['time_stepping']['t_max']
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'inverse_w'
    #print " "
    #print "namelist"
    #print pp.pprint(setup["namelist"])
    #print " "
    #print "paramlist"
    #print pp.pprint(setup["paramlist"])

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])
    
    # simulation results 
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_GATE_III(sim_data):
    """
    plot GATE_III profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 100)

    pls.plot_mean(data_to_plot,   "GATE_III_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "GATE_III_quicklook_drafts.pdf")

def test_plot_timeseries_GATE_III(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "GATE_III")

