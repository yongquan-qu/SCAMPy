import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import os
import subprocess
from pathlib import Path

from netCDF4 import Dataset

import pytest

import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # remove netcdf file from previous failed test
    request.addfinalizer(cmn.removing_files)
    # generate namelists and paramlists
    setup = cmn.simulation_setup('DYCOMS_RF01')

    #setup['namelist']['microphysics']['rain_model'] = 'cutoff'
    setup['namelist']['microphysics']['rain_model'] = 'clima_1m'

    #setup['namelist']['thermodynamics']['sgs'] = 'quadrature'
    #setup["namelist"]["turbulence"]["EDMF_PrognosticTKE"]["entrainment"]="moisture_deficit"

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf files after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_DYCOMS_RF01_drizzle(sim_data):
    """
    plot drizzling DYCOMS_RF01 timeseries
    """
    # make directory
    localpath = Path.cwd()
    (localpath / "plots/output/DYCOMS_RF01_drizzle/all_variables/").mkdir(parents=True, exist_ok=True)
    les_data_path = localpath / "les_data/DYCOMS_RF01.nc"
    if not les_data_path.is_file():
        url_ = r"https://drive.google.com/uc?export=download&id=1qBpJRVZsaQeJLuBEUwzLDv0aNlZsBaEq"
        os.system(f"curl -sLo {les_data_path} '{url_}'")
    les_data = Dataset(les_data_path, 'r')

    f1 = "plots/output/DYCOMS_RF01_drizzle/"
    f2 = f1 + "all_variables/"
    cn = "DYCOMS_RF01_drizzle_"
    t0 = 3
    t1 = 4
    zmin = 0.0
    zmax = 1.2
    cb_min = [0., 0.]
    cb_max = [0.9, 1.4]
    fixed_cbar = True
    cb_min_t = [287.5, 287.5, 288.5, 0, 0, 0, -1, -1,0, -1, 0, 0, 0, 9,\
                -0.16, 0, 4.2, -5.5,\
                 0, -0.25, 0,\
                -0.06, -0.1, -0.1,\
                 0., -0.05, -1e-5]
    cb_max_t = [307.5, 307.5, 289.5, 0.9, 0.9, 1.5, 4e-5, 4e-5, 100, 4e-5, 100, 12, 12, 11,\
                0, 1.4, 7.2, -3.5,\
                0.24, 0.05, 1.5,\
                0.02, 0.15, 0.15,\
                0.05, 0.008, 0.0001]

    scm_dict = cmn.read_scm_data(sim_data)
    les_dict = cmn.read_les_data(les_data)

    scm_dict_t = cmn.read_scm_data_timeseries(sim_data)
    les_dict_t = cmn.read_les_data_timeseries(les_data)

    pls.plot_closures(scm_dict, les_dict, t0, t1, zmin, zmax, cn+"closures.pdf", folder=f1)
    pls.plot_spec_hum(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"humidities.pdf", folder=f1)
    pls.plot_upd_prop(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"updraft_properties.pdf", folder=f1)
    pls.plot_fluxes(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"mean_fluxes.pdf", folder=f1)
    pls.plot_tke_comp(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"tke_components.pdf", folder=f1)

    pls.plot_cvar_mean(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"var_covar_mean.pdf", folder=f2)
    pls.plot_cvar_comp(scm_dict, t0, t1, zmin, zmax,  cn+"var_covar_components.pdf", folder=f2)
    pls.plot_tke_break(scm_dict, les_dict, t0, t1, zmin, zmax, cn+"tke_breakdown.pdf",folder=f2)

    pls.plot_contour_t(scm_dict, les_dict, fixed_cbar, cb_min_t, cb_max_t, zmin, zmax, folder=f2)
    pls.plot_mean_prof(scm_dict, les_dict, t0, t1,  zmin, zmax, folder=f2)

    pls.plot_main(scm_dict_t, les_dict_t, scm_dict, les_dict,
                  cn+"main_timeseries.pdf", cb_min, cb_max, zmin, zmax, folder=f1)

    pls.plot_1D(scm_dict_t, les_dict_t, cn, folder=f2)


