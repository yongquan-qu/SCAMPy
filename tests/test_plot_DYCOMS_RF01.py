import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./tests")

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
    setup = pls.simulation_setup('DYCOMS_RF01')
    # chenge the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_scalar_var'] = False

    #setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

    setup['paramlist']['turbulence']['updraft_microphysics']['max_supersaturation'] = 100.      #0.1      # 0.1
    #TODO sa_quadrature + similarity_diff + calc covar doesnt work -> self.wstar in Turbulence.pyx line 134 is zero (division by zero)

    #                                                                                           #best     # default
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1                #0.25     # 0.1
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.05               #0.075    # 0.5    <---- /10
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 10.              #20       # 0.01   <---- *1000
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] = 0.375       #0.375    # 0.375
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] = 5.       #5        # 500.0  <---- /100

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf files after tests
    #request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 quicklook profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 100)

    pls.plot_mean(data_to_plot,   "DYCOMS_RF01_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "DYCOMS_RF01_quicklook_drafts.pdf")

def test_plot_var_covar_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 quicklook profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, 100)

    pls.plot_var_covar_mean(data_to_plot,   "DYCOMS_RF01_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,   "DYCOMS_RF01_var_covar_components.pdf")

def test_plot_timeseries_DYCOMS(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "DYCOMS")

def test_DYCOMS_RF01_radiation(sim_data):
    """
    - check if the initial radiative flux is the same as in the reference simulation
    - do quicklook plots of radiation forcing (init and final)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt_data = pls.read_data_avg(sim_data,     100)
    rad_data = pls.read_rad_data_avg(sim_data, 100)

    # plot
    mpl.rc('lines', linewidth=2, markersize=8)

    plt.figure(1, figsize=(18,14))
    plots = []
    # loop over simulation and reference data for t=0 and t=-1
    x_lab  = ['longwave radiative flux [W/m2]', 'dTdt [K/day]',       'QT [g/kg]',         'QL [g/kg]']
    legend = ["lower right",                    "lower left",         "lower left",        "lower right"]
    line   = ['--',                             '--',                 '-',                 '-']
    plot_y = [rad_data["rad_flux"],             rad_data["rad_dTdt"], plt_data["qt_mean"], plt_data["ql_mean"]]
    plot_x = [rad_data["z"],                    plt_data["z_half"],   plt_data["z_half"],  plt_data["z_half"]]
    color  = ["palegreen",                      "forestgreen"]
    label  = ["ini",                            "end"        ]

    for plot_it in xrange(4):
        plots.append(plt.subplot(2,2,plot_it+1))
                              #(rows, columns, number)
        for it in xrange(2):
            plots[plot_it].plot(plot_y[plot_it][it], plot_x[plot_it], '.-', color=color[it], label=label[it])
        plots[plot_it].legend(loc=legend[plot_it])
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
    plots[2].set_xlim([1, 10])
    plots[3].set_xlim([-0.1, 0.5])

    plt.savefig("tests/output/DYCOMS_RF01_radiation.pdf")
    plt.clf()
