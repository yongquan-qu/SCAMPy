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
    setup = cmn.simulation_setup('DYCOMS_RF01')

    # run scampy
    #subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    #scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf files after tests
    #request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01/")
    except:
        print('DYCOMS_RF01 folder exists')
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01/all_variables/")
    except:
        print('DYCOMS_RF01/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/DYCOMS_RF01.nc")):
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/dh636h4owlt6a79/DYCOMS_RF01.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/DYCOMS_RF01.nc "+url_)
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')

    f1 = "plots/output/DYCOMS_RF01/"
    f2 = f1 + "all_variables/"
    cn = "DYCOMS_RF01_"
    t0 = 3
    t1 = 4
    cb_min = [0., 0.]
    cb_max = [0.9, 1.4]

    scm_dict = cmn.read_scm_data(sim_data)
    les_dict = cmn.read_les_data(les_data)

    scm_dict_t = cmn.read_scm_data_timeseries(sim_data)
    les_dict_t = cmn.read_les_data_timeseries(les_data)

    pls.plot_closures(scm_dict, les_dict, t0, t1, cn+"closures.pdf", folder=f1)
    pls.plot_spec_hum(scm_dict, les_dict, t0, t1, cn+"humidities.pdf", folder=f1)
    pls.plot_upd_prop(scm_dict, les_dict, t0, t1, cn+"updraft_properties.pdf", folder=f1)
    pls.plot_tke_comp(scm_dict, les_dict, t0, t1, cn+"tke_components.pdf", folder=f1)

    pls.plot_cvar_mean(scm_dict, les_dict, t0, t1, cn+"var_covar_mean.pdf", folder=f2)
    pls.plot_cvar_comp(scm_dict, t0, t1, cn+"var_covar_components.pdf", folder=f2)
    pls.plot_tke_break(scm_dict, les_dict, t0, t1, cn+"tke_breakdown.pdf",folder=f2)

    pls.plot_contour_t(scm_dict, les_dict, folder=f2)
    pls.plot_mean_prof(scm_dict, les_dict, t0, t1, folder=f2)

    pls.plot_main(scm_dict_t, les_dict_t, scm_dict, les_dict,
                  cn+"main_timeseries.pdf", cb_min, cb_max, folder=f1)

    pls.plot_1D(scm_dict_t, les_dict_t, folder=f2)
