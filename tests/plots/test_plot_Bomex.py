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

import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    cmn.removing_files
    setup = cmn.simulation_setup('Bomex')
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
def test_plot_Bomex(sim_data):
    """
    plot Bomex profiles
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc', 'r')
    data_to_plot = cmn.read_data_avg(sim_data, tmin=1)
    les_data_to_plot = cmn.read_les_data_avg(les_data, tmin=1)

    pls.plot_mean(data_to_plot,  les_data_to_plot,   "Bomex_quicklook.pdf")
    pls.plot_drafts(data_to_plot, les_data_to_plot,  "Bomex_quicklook_drafts.pdf")
    pls.plot_closures(data_to_plot, les_data_to_plot,  "Bomex_closures.pdf")
    pls.plot_velocities(data_to_plot, les_data_to_plot,  "Bomex_velocities.pdf")

def test_plot_timeseries_Bomex(sim_data):
    """
    plot Bomex timeseries
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          "Bomex")
    pls.plot_mean(data_to_plot, les_data_to_plot,5,6,            "Bomex_quicklook.pdf")
    pls.plot_closures(data_to_plot, les_data_to_plot,5,6,        "Bomex_closures.pdf")
    pls.plot_drafts(data_to_plot, les_data_to_plot,5,6,          "Bomex_quicklook_drafts.pdf")
    pls.plot_velocities(data_to_plot, les_data_to_plot,5,6,      "Bomex_velocities.pdf")
    pls.plot_tapio(data_to_plot, les_data_to_plot,5,6,           "Bomex_main.pdf")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 5,6, "Bomex_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,5,6,              "Bomex_var_covar_components.pdf")

def test_plot_timeseries_1D_Bomex(sim_data):
    """
    plot Bomex 1D timeseries
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot, "Bomex_timeseries_1D.pdf")

@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_var_covar_Bomex(sim_data):
    """
    plot Bomex var covar
    """
    les_data = Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc', 'r')
    data_to_plot = cmn.read_data_avg(sim_data, tmin=1, var_covar=True)
    les_data_to_plot = cmn.read_les_data_avg(les_data, tmin=1, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,  les_data_to_plot,     "Bomex_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "Bomex_var_covar_components.pdf")
