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
    setup = cmn.simulation_setup('ARM_SGP')

    # change the defaults if needed
    # setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    #subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    #scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    #request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_timeseries_ARM_SGP(sim_data):
    """
    plot ARM_SGP timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/")
    except:
        print('ARM_SGP folder exists')
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/all_variables/")
    except:
        print('ARM_SGP/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/ARM_SGP.nc")):
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/j7s0jmmkwn7av62/ARM_SGP.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/ARM_SGP.nc "+url_)
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')

    scm_data_to_plot = cmn.read_scm_data(sim_data)
    les_data_to_plot = cmn.read_les_data(les_data)
    f1 = "plots/output/ARM_SGP/"
    f2 = f1 + "all_variables/"
    cn = "ARM_SGP_"
    t0 = 8
    t1 = 11

    pls.plot_closures(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"closures.pdf", folder=f1)
    pls.plot_humidities(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"humidities.pdf", folder=f1)
    pls.plot_updraft_properties(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"updraft_properties.pdf", folder=f1)
    pls.plot_tke_components(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"tke_components.pdf", folder=f1)

    pls.plot_var_covar_mean(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"var_covar_mean.pdf", folder=f2)
    pls.plot_var_covar_components(scm_data_to_plot, t0, t1, cn+"var_covar_components.pdf", folder=f2)
    pls.plot_tke_breakdown(scm_data_to_plot, les_data_to_plot, t0, t1, cn+"tke_breakdown.pdf", folder=f2)

    pls.plot_contour_timeseries(scm_data_to_plot, les_data_to_plot, folder=f2)
    pls.plot_mean(scm_data_to_plot, les_data_to_plot, t0, t1, folder=f2)

def test_plot_timeseries_1D_ARM_SGP(sim_data):
    """
    plot ARM_SGP 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/")
        print()
    except:
        print('ARM_SGP folder exists')
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/all_variables/")
    except:
        print('ARM_SGP/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/ARM_SGP.nc")):
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/j7s0jmmkwn7av62/ARM_SGP.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/ARM_SGP.nc "+url_)
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')

    scm_data_to_plot_timeseries = cmn.read_scm_data_timeseries(sim_data)
    les_data_to_plot_timeseries = cmn.read_les_data_timeseries(les_data)

    scm_data_to_plot = cmn.read_scm_data(sim_data)
    les_data_to_plot = cmn.read_les_data(les_data)

    pls.plot_main_timeseries(scm_data_to_plot_timeseries,
                             les_data_to_plot_timeseries,
                             scm_data_to_plot,
                             les_data_to_plot,
                             "ARM_SGP_main_timeseries.pdf",
                             folder="plots/output/ARM_SGP/")

    pls.plot_timeseries_1D(scm_data_to_plot_timeseries,
                           les_data_to_plot_timeseries,
                           folder="plots/output/ARM_SGP/all_variables/")
