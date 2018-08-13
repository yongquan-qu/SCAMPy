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
    setup = pls.simulation_setup('Rico')
    # change the defaults
    #setup["namelist"]['stats_io']['frequency'] = setup["namelist"]['time_stepping']['t_max']
    #setup["namelist"]['time_stepping']['t_max'] = 3*60*60
    #setup["namelist"]['time_stepping']['dt'] = 1.
    #setup["namelist"]['stats_io']['frequency'] = 60
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['use_scalar_var'] = True

    #setup['namelist']['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 3

    #setup['namelist']['turbulence']['sgs'] = {}
    #setup['namelist']['turbulence']['sgs']['use_prescribed_scalar_var'] = True
    #setup['namelist']['turbulence']['sgs']['prescribed_QTvar'] = 1e-8
    #setup['namelist']['turbulence']['sgs']['prescribed_Hvar'] = 1e-3
    #setup['namelist']['turbulence']['sgs']['prescribed_HQTcov'] = -1e-4
    setup['paramlist']['turbulence']['updraft_microphysics']['max_supersaturation'] = .005

    #setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

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
    #request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_Rico(sim_data):
    """
    plot Rico profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 10)

    pls.plot_mean(data_to_plot,   "Rico_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "Rico_quicklook_drafts.pdf")

def test_plot_var_covar_Rico(sim_data):
    """
    plot Rico variance and covariance of H and QT profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 10)

    pls.plot_var_covar_mean(data_to_plot,   "Rico_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,   "Rico_var_covar_components.pdf")

def test_plot_timeseries_Rico(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Rico")

