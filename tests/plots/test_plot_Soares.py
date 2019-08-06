import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import numpy as np
import subprocess
import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = cmn.simulation_setup('Soares')
    # change the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_Soares(sim_data):
    """
    plot Soares profiles
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/Soares/stats/Stats.Soares.nc', 'r')
    data_to_plot = cmn.read_data_avg(sim_data, tmin=1)
    les_data_to_plot = cmn.read_les_data_avg(les_data, tmin=1)

    pls.plot_mean(data_to_plot,les_data_to_plot,   "Soares_quicklook.pdf")
    pls.plot_drafts(data_to_plot,les_data_to_plot, "Soares_quicklook_drafts.pdf")
    pls.plot_closures(data_to_plot, les_data_to_plot,  "Soares_closures.pdf")
    pls.plot_velocities(data_to_plot, les_data_to_plot,  "Soares_velocities.pdf")

def test_plot_timeseries_Soares(sim_data):
    """
    plot Soares timeseries
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/Soares/stats/Stats.Soares.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    # pls.plot_timeseries(data_to_plot,les_data_to_plot, "Soares")
    # pls.plot_tapio(data_to_plot, les_data_to_plot,5,7, "Soares_main.pdf")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          "Soares")
    pls.plot_mean(data_to_plot, les_data_to_plot,7,8,            "Soares_quicklook.pdf")
    pls.plot_closures(data_to_plot, les_data_to_plot,7,8,        "Soares_closures.pdf")
    pls.plot_drafts(data_to_plot, les_data_to_plot,7,8,          "Soares_quicklook_drafts.pdf")
    pls.plot_velocities(data_to_plot, les_data_to_plot,7,8,      "Soares_velocities.pdf")
    pls.plot_tapio(data_to_plot, les_data_to_plot,7,8,           "Soares_main.pdf")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 7,8, "Soares_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,7,8,              "Soares_var_covar_components.pdf")

def test_plot_timeseries_1D_Soares(sim_data):
    """
    plot Soares 1D timeseries
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/Soares/stats/Stats.Soares.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot,les_data_to_plot, "Soares_timeseries_1D.pdf")

@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_var_covar_Soares(sim_data):
    """
    plot Soares var covar profiles
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/Soares/stats/Stats.Soares.nc', 'r')
    data_to_plot = cmn.read_data_avg(sim_data, tmin=1, var_covar=True)
    les_data_to_plot = cmn.read_les_data_avg(les_data, tmin=1, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot,   "Soares_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "Soares_var_covar_comp.pdf")
