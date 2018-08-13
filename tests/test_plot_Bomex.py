import sys
sys.path.insert(0, "./")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import numpy as np
import pprint as pp

import main as scampy
import plot_scripts as pls
import pytest_wrapper as wrp

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('Bomex')
    # change the defaults
    #setup["namelist"]['time_stepping']['t_max'] = 15000
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'  # dry, inverse_w, b_w2
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['use_scalar_var'] = True

    #setup['namelist']['turbulence']['sgs'] = {}
    #setup['namelist']['turbulence']['sgs']['use_prescribed_scalar_var'] = False
    #setup['namelist']['turbulence']['sgs']['prescribed_QTvar'] = 0.5 * 1e-7
    #setup['namelist']['turbulence']['sgs']['prescribed_Hvar'] = 0.01
    #setup['namelist']['turbulence']['sgs']['prescribed_HQTcov'] = -1e-3

    #setup['paramlist']['turbulence']['updraft_microphysics']['max_supersaturation'] = 100. #0.1

    #setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

    #TODO - use_local_micro=False    - no clouds
    #     - entrainment = bw_2       - oscillations in w
    #     - entrainment = inwerse_v  - random ql and cloud fraction

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    #request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_Bomex(sim_data):
    """
    plot Bomex profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 100)

    pls.plot_mean(data_to_plot,   "Bomex_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "Bomex_quicklook_drafts.pdf")

def test_plot_timeseries_Bomex(sim_data):
    """
    plot Bomex timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Bomex")

def test_plot_var_covar_Bomex(sim_data):
    """
    plot Bomaex var covar
    """
    data_to_plot = pls.read_data_avg(sim_data, 100)

    pls.plot_var_covar_mean(data_to_plot,   "Bomex_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,   "Bomex_var_covar_components.pdf")

def test_plot_Bomex_fig3(sim_data, folder="tests/output/"):
    """
    Plot Fig3 from Bomex papaer
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    title = "BOMEX_fig3.pdf"
    plt_data = pls.read_data_avg(sim_data, 100)

    mean_tke    = plt_data["tke_mean"][1]
    mean_tke_ini= plt_data["tke_mean"][0]
    udr_ar      = plt_data["updraft_area"][1]
    udr_ar_ini  = plt_data["updraft_area"][0]
    mean_u      = plt_data["u_mean"][1]
    mean_u_ini  = plt_data["u_mean"][0]
    mean_v      = plt_data["v_mean"][1]
    mean_v_ini  = plt_data["v_mean"][0]
    mean_qv     = plt_data["qt_mean"][1] - plt_data["ql_mean"][1]
    mean_qv_ini = plt_data["qt_mean"][0] - plt_data["ql_mean"][0]
    mean_ql     = plt_data["ql_mean"][1]
    mean_ql_ini = plt_data["ql_mean"][0]
    p0          = np.array(sim_data["reference/p0"][:])
    mean_th     = np.zeros(p0.size)
    mean_th_ini = np.zeros(p0.size)
    for it in range(p0.size):
      mean_th[it]     = wrp.theta_c(p0[it], plt_data["temperature_mean"][1][it])
      mean_th_ini[it] = wrp.theta_c(p0[it], plt_data["temperature_mean"][0][it])

    plt.figure(1, figsize=(18,14))
    mpl.rc('lines', linewidth=4, markersize=10)
    mpl.rcParams.update({'font.size': 18})
    plots = []
    # iteration over plots
    x_lab      = ['TH [K]',     'QV [g/kg]', 'U[m/s]; V[m/s]', 'QL [g/kg]', 'TKE',        'updraft area [%]']
    plot_x     = [ mean_th,     mean_qv,     mean_u,           mean_ql,     mean_tke,     udr_ar*100 ]
    plot_x_ini = [ mean_th_ini, mean_qv_ini, mean_u_ini,       mean_ql_ini, mean_tke_ini, udr_ar_ini*100  ]

    for plot_it in range(6):
        plots.append(plt.subplot(3,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_ylim([0, 2500])
        plots[plot_it].grid(True)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].plot(plot_x_ini[plot_it], plt_data["z_half"], '.-', label="ini")
        plots[plot_it].plot(plot_x[plot_it],     plt_data["z_half"], '.-', label="end")
        if (plot_it == 2):
            plots[plot_it].plot(mean_v_ini, plt_data["z_half"], '.-', label="ini")
            plots[plot_it].plot(mean_v,     plt_data["z_half"], '.-', label="end")

    plots[2].legend(loc='upper left')
    plots[0].set_xlim([298, 310])
    plots[1].set_xlim([4, 18])
    plots[2].set_xlim([-10, 2.0])
    #plots[3].set_xlim([0, 0.02])
    plt.savefig(folder + title)
    plt.clf()
