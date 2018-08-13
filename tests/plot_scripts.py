import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./tests")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset
import numpy as np
import pprint as pp

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker

def simulation_setup(case):
    """
    generate namelist and paramlist files for scampy
    choose the name of the output folder
    """
    # Filter annoying Cython warnings that serve no good purpose.
    # see https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    # simulation related parameters
    os.system("python generate_namelist.py " + case)
    file_case = open(case + '.in').read()
    # turbulence related parameters
    os.system("python generate_paramlist.py " +  case)
    file_params = open('paramlist_' + case + '.in').read()

    namelist  = json.loads(file_case)
    paramlist = json.loads(file_params)

    namelist['output']['output_root'] = "./Tests."
    namelist['meta']['uuid'] = case
    # TODO - copied from NetCDFIO
    # ugly way to know the name of the folder where the data is saved
    uuid = str(namelist['meta']['uuid'])
    outpath = str(
        os.path.join(
            namelist['output']['output_root'] +
            'Output.' +
            namelist['meta']['simname'] +
            '.' +
            uuid[len(uuid )-5:len(uuid)]
        )
    )
    outfile = outpath + "/stats/Stats." + case + ".nc"

    res = {"namelist"  : namelist,
           "paramlist" : paramlist,
           "outfile"   : outfile}
    return res


def removing_files():
    """
    remove the folder with netcdf files from tests
    """
    #TODO - think of something better
    cmd = "rm -r Tests.Output.*"
    subprocess.call(cmd , shell=True)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def read_data_avg(sim_data, n_steps):
    """
    Read in the data from netcdf file into a dictionary that can be used for quicklook plots
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w",\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain"
                ]

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:])}

    time = [0, -1]
    for var in variables:
        data_to_plot[var] = []
        for it in xrange(2):
            if ("buoyancy" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 10000) #cm2/s3
            elif ("qt" in var or "ql" in var or "qr" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 1000)  #g/kg
            elif ("p0" in var):
                data_to_plot[var].append(np.array(sim_data["reference/" + var][time[it], :]) * 100)  #hPa
            else:
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    #if(n_steps > 0):
    #    for var in variables:
    #        for time_it in xrange(-2, -1*n_steps-1, -1):
    #            if ("buoyancy" in var):
    #                data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 10000  #cm2/s3
    #            elif ("qt" in var or "ql" in var or "qr" in var):
    #                data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 1000   #g/kg
    #            elif ("p0" in var):
    #                data_to_plot[var][1] += np.array(sim_data["reference/" + var][time_it, :]) * 100   #hPa
    #            else:
    #                data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

    #        data_to_plot[var][1] /= n_steps

    return data_to_plot


def read_rad_data_avg(sim_data, n_steps):
    """
    Read in the radiation forcing data from netcdf files into a dictionary that can be used for quicklook plots
    """
    variables = ["rad_flux", "rad_dTdt"]

    time = [0, -1]
    rad_data = {"z" : np.array(sim_data["profiles/z"][:])}
    for var in variables:
        rad_data[var] = []
        for it in xrange(2):
            if ("rad_dTdt" in var):
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 60 * 60 * 24) # K/day
            else:
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    #if(n_steps > 0):
    #    for var in variables:
    #        for time_it in xrange(-2, -1*n_steps-1, -1):
    #            if ("rad_dTdt" in var):
    #                rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :] * 60 * 60 * 24) # K/day
    #            else:
    #                rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

    #        rad_data[var][1] /= n_steps

    return rad_data

def read_data_srs(sim_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for quicklook timeseries plots
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w",\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain"\
                ]

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        if ("buoyancy" in var):
            data_to_plot[var] = np.array(sim_data["profiles/"  + var][:, :]) * 10000 #cm2/s3
        elif ("qt" in var or "ql" in var or "qr" in var):
            data_to_plot[var] = np.array(sim_data["profiles/"  + var][:, :]) * 1000  #g/kg
        elif ("p0" in var):
            data_to_plot[var] = np.array(sim_data["reference/" + var][:, :]) * 100   #hPa
        else:
            data_to_plot[var] = np.array(sim_data["profiles/"  + var][:, :])

    return data_to_plot

def plot_mean(data, title, folder="tests/output/"):
    """
    Plots mean profiles from Scampy (current test run and reference simulation - EDMF_BulkSteady)
    """
    # customize defaults
    mpl.rc('lines', linewidth=3, markersize=8)

    plt.figure(1, figsize=(18,14))
    mpl.rc('lines', linewidth=4, markersize=10)
    mpl.rcParams.update({'font.size': 18})
    plots = []
    qv_mean = np.array(data["qt_mean"]) - np.array(data["ql_mean"])
    # iteration over plots
    x_lab  = ['QV [g/kg]', 'QL [g/kg]',      'QR [g/kg]',      'THL [K]',           'buoyancy [cm2/s3]',   'TKE [m2/s2]']
    plot_x = [qv_mean,      data["ql_mean"],  data["qr_mean"], data["thetal_mean"],  data["buoyancy_mean"], data["tke_mean"]]
    color  = ["palegreen", "forestgreen"]
    label  = ["ini", "end"]

    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        for it in xrange(2): #init, end
            plots[plot_it].plot(plot_x[plot_it][it], data["z_half"], '.-', color=color[it], label=label[it])

    plots[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_drafts(data, title, folder="tests/output/"):
    """
    Plots updraft and environment profiles from Scampy (current and reference simulation - EDMF_Bulksteady)
    """
    # customize defaults
    mpl.rc('lines', linewidth=3, markersize=8)

    plt.figure(1, figsize=(18,14))
    mpl.rc('lines', linewidth=4, markersize=10)
    mpl.rcParams.update({'font.size': 18})
    plots = []
    #qv_mean    = np.array(data["qt_mean"])    - np.array(data["ql_mean"])
    #env_qv     = np.array(data["env_qt"])     - np.array(data["env_ql"])
    #updraft_qv = np.array(data["updraft_qt"]) - np.array(data["updraft_ql"])
    # iteration over plots
    #x_lab    = ["QV [g/kg]", "QL [g/kg]",        "QR [g/kg]",        "w [m/s]",         "updraft buoyancy [cm2/s3]",  "updraft area [%]"]
    x_lab    = ["QT [g/kg]", "QL [g/kg]",        "QR [g/kg]",        "w [m/s]",         "updraft buoyancy [cm2/s3]",  "updraft area [%]"]
    #plot_upd = [qv_mean,     data["updraft_ql"], data["updraft_qr"], data["updraft_w"], data["updraft_buoyancy"],     data["updraft_area"]]
    #plot_env = [env_qv,      data["env_ql"],     data["env_qr"],     data["env_w"]]
    #plot_mean= [updraft_qv,  data["ql_mean"],    data["qr_mean"]]
    plot_upd = [data["qt_mean"],     data["updraft_ql"], data["updraft_qr"], data["updraft_w"], data["updraft_buoyancy"],     data["updraft_area"]]
    plot_env = [data["env_qt"],      data["env_ql"],     data["env_qr"],     data["env_w"]]
    plot_mean= [data["updraft_qt"],  data["ql_mean"],    data["qr_mean"]]
    color_mean= "purple"
    color_env = "red"
    color_upd = "blue"
    label_mean= "mean"
    label_env = "env"
    label_upd = "upd"

    for plot_it in xrange(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        #plot updrafts
        if (plot_it != 5):
            plots[plot_it].plot(plot_upd[plot_it][1], data["z_half"], ".-", color=color_upd, label=label_upd)
        if (plot_it == 5):
            plots[plot_it].plot(plot_upd[plot_it][1] * 100, data["z_half"], ".-", color=color_upd, label=label_upd)
        # plot environment
        if (plot_it < 4):
            plots[plot_it].plot(plot_env[plot_it][1], data["z_half"], ".-", color=color_env, label=label_env)
        # plot mean
        if (plot_it < 3):
            plots[plot_it].plot(plot_mean[plot_it][1], data["z_half"], ".-", color=color_mean, label=label_mean)


    plots[0].legend(loc='upper right')
    plt.savefig(folder + title)
    plt.clf()

def plot_var_covar_mean(data, title, folder="tests/output/"):
    """
    Plots variance and covariance profiles from Scampy
    """
    # customize defaults
    mpl.rc('lines', linewidth=3, markersize=8)
    plt.figure(1, figsize=(18,14))
    mpl.rc('lines', linewidth=4, markersize=10)
    mpl.rcParams.update({'font.size': 18})
    plots = []

    plot_Hvar_m   = ["Hvar_mean", "env_Hvar"]
    plot_QTvar_m  = ["QTvar_mean",  "env_QTvar"]
    plot_HQTcov_m = ["HQTcov_mean", "env_HQTcov"]
    color_m  = ['black', 'red']
    color_m0 = ['gray', 'orange']
    x_lab = ["Hvar", "QTvar", "HQTcov"]

    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

    for var in range(2):
        plots[0].plot(data[plot_Hvar_m[var]][0],   data["z_half"], ".-", label=plot_Hvar_m[var],  c=color_m0[var])
        plots[0].plot(data[plot_Hvar_m[var]][1],   data["z_half"], ".-", label=plot_Hvar_m[var],  c=color_m[var])
        plots[1].plot(data[plot_QTvar_m[var]][0],  data["z_half"], ".-", label=plot_QTvar_m[var], c=color_m0[var])
        plots[1].plot(data[plot_QTvar_m[var]][1],  data["z_half"], ".-", label=plot_QTvar_m[var], c=color_m[var])
        plots[2].plot(data[plot_HQTcov_m[var]][0], data["z_half"], ".-", label=plot_HQTcov_m[var],c=color_m0[var])
        plots[2].plot(data[plot_HQTcov_m[var]][1], data["z_half"], ".-", label=plot_HQTcov_m[var],c=color_m[var])

    #plots[0].set_xlim([0, 1e-3])
    #plots[1].set_xlim([0, 1e-7])
    #plots[2].set_xlim([0, 1e-5])

    plots[0].axvline(0.01,       c='green', label='assumed')
    plots[1].axvline(0.5 * 1e-7, c='green', label='assumed')
    plots[2].axvline(-1e-3,      c='green', label='assumed')
    plots[0].axhline(1500, c='green')
    plots[0].axhline(500,  c='green')
    plots[1].axhline(1500, c='green')
    plots[2].axhline(1500, c='green')
    plots[2].axhline(200,  c='green')

    plots[0].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_var_covar_components(data, title, folder="tests/output/"):
    """
    Plots variance and covariance profiles from Scampy
    """
    plots = []
    plot_Hvar_c   = ["Hvar_dissipation", "Hvar_entr_gain", "Hvar_detr_loss", "Hvar_shear", "Hvar_rain"]
    plot_QTvar_c  = ["QTvar_dissipation", "QTvar_entr_gain", "QTvar_detr_loss", "QTvar_shear", "QTvar_rain"]
    plot_HQTcov_c = ["HQTcov_dissipation","HQTcov_entr_gain", "HQTcov_detr_loss", "HQTcov_shear", "HQTcov_rain"]

    color_c = ['green', 'pink', 'purple', 'orange', 'blue']
    x_lab = ["Hvar", "QTvar", "HQTcov"]

    # customize defaults
    fig = plt.figure(1, figsize=(18,14))
    mpl.rc('lines', linewidth=6, markersize=12)
    mpl.rcParams.update({'font.size': 20})

    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

    for var in range(5):
        plots[0].plot(data[plot_Hvar_c[var]][1],   data["z_half"], ".-", label=plot_Hvar_c[var],  c=color_c[var])
        plots[1].plot(data[plot_QTvar_c[var]][1],  data["z_half"], ".-", label=plot_QTvar_c[var], c=color_c[var])
        plots[2].plot(data[plot_HQTcov_c[var]][1], data["z_half"], ".-", label=plot_HQTcov_c[var],c=color_c[var])

    plots[0].axhline(1500, c='gray')
    plots[0].axhline(500,  c='gray')
    plots[1].axhline(1500, c='gray')
    plots[2].axhline(1500, c='gray')
    plots[2].axhline(200,  c='gray')

    plots[0].legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(folder + title)
    plt.clf()

def plot_timeseries(plt_data, case):

    output_folder="tests/output/"
    mpl.rcParams.update({'font.size': 18})

    z_half  = plt_data["z_half"]
    time    = plt_data["t"] / 60. / 60.

    mean_ql  = np.transpose(plt_data["ql_mean"])
    mean_qr  = np.transpose(plt_data["qr_mean"])
    mean_qt  = np.transpose(plt_data["qt_mean"])
    mean_qv  = mean_qt - mean_ql
    mean_tke = np.transpose(plt_data["tke_mean"])
    mean_buo = np.transpose(plt_data["buoyancy_mean"])

    updr_buo  = np.transpose(plt_data["updraft_buoyancy"])
    updr_qt   = np.transpose(plt_data["updraft_qt"])
    updr_ql   = np.transpose(plt_data["updraft_ql"])
    updr_qr   = np.transpose(plt_data["updraft_qr"])
    updr_qv   = updr_qt - updr_ql
    updr_w    = np.transpose(plt_data["updraft_w"])
    updr_area = np.transpose(plt_data["updraft_area"])

    #print " "
    #print "updraft_w range: ", np.min(plt_data["updraft_w"]), " -- ", np.max(plt_data["updraft_w"])
    #print "after transpose: ", np.min(updr_w), " -- ", np.max(updr_w)

    env_qt = np.transpose(plt_data["env_qt"])
    env_ql = np.transpose(plt_data["env_ql"])
    env_qr = np.transpose(plt_data["env_qr"])
    env_qv = env_qt - env_ql
    env_w  = np.transpose(plt_data["env_w"])

    fig = plt.figure(1)
    fig.set_figheight(16)
    fig.set_figwidth(28)
    ax = []
    for plot_it in range(6):
        ax.append(fig.add_subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        ax[plot_it].set_xlabel('t [hrs]')
        ax[plot_it].set_ylabel('z [m]')

    #plot0 = ax[0].pcolormesh(time, z_half, mean_qt,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot0 = ax[0].pcolormesh(time, z_half, mean_qt, rasterized=True)
    fig.colorbar(plot0, ax=ax[0], label='mean qt [g/kg]')
    #plot1 = ax[1].pcolormesh(time, z_half, mean_buo, cmap=discrete_cmap(8,mpl.cm.bone_r),rasterized=True)
    plot1 = ax[1].pcolormesh(time, z_half, mean_buo,rasterized=True)
    fig.colorbar(plot1, ax=ax[1], label='mean buo [cm2/s3]')
    #plot2 = ax[2].pcolormesh(time, z_half, mean_tke, cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True) #, vmin=0, vmax=5)
    plot2 = ax[2].pcolormesh(time, z_half, mean_tke, rasterized=True) #, vmin=0, vmax=5)
    fig.colorbar(plot2, ax=ax[2], label='mean tke [m2/s2]')
    #plot3 = ax[3].pcolormesh(time, z_half, mean_qv,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot3 = ax[3].pcolormesh(time, z_half, mean_qv, rasterized=True)
    fig.colorbar(plot3, ax=ax[3], label='mean qv [g/kg]')
    #plot4 = ax[4].pcolormesh(time, z_half, mean_ql, cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot4 = ax[4].pcolormesh(time, z_half, mean_ql, rasterized=True)
    fig.colorbar(plot4, ax=ax[4], label='mean ql [g/kg]')
    #plot5 = ax[5].pcolormesh(time, z_half, mean_qr,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot5 = ax[5].pcolormesh(time, z_half, mean_qr, rasterized=True)
    fig.colorbar(plot5, ax=ax[5], label='mean qr [g/kg]')
    plt.savefig(output_folder + case + "_timeseries_01mean.pdf")
    plt.clf()

    fig = plt.figure(1)
    ax = []
    for plot_it in range(6):
        ax.append(fig.add_subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        ax[plot_it].set_xlabel('t [hrs]')
        ax[plot_it].set_ylabel('z [m]')

    plot0 = ax[0].pcolormesh(time, z_half, env_qt, rasterized=True)
    fig.colorbar(plot0, ax=ax[0], label='env qt [g/kg]')
    plot1 = ax[1].pcolormesh(time, z_half, (1.-updr_area)*100, rasterized=True)
    fig.colorbar(plot1, ax=ax[1], label='env area [%]')
    #plot0 = ax[0].pcolormesh(time, z_half, env_w,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot2 = ax[2].pcolormesh(time, z_half, env_w, rasterized=True)
    fig.colorbar(plot2, ax=ax[2], label='env w [m/2]')
    #plot3 = ax[3].pcolormesh(time, z_half, env_qv,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot3 = ax[3].pcolormesh(time, z_half, env_qv, rasterized=True)
    fig.colorbar(plot3, ax=ax[3], label='env qv [g/kg]')
    #plot4 = ax[4].pcolormesh(time, z_half, env_ql,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot4 = ax[4].pcolormesh(time, z_half, env_ql, rasterized=True)
    fig.colorbar(plot4, ax=ax[4], label='env ql [g/kg]')
    #plot5 = ax[5].pcolormesh(time, z_half, env_qr,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot5 = ax[5].pcolormesh(time, z_half, env_qr, rasterized=True)
    fig.colorbar(plot5, ax=ax[5], label='env qr [g/kg]')
    plt.savefig(output_folder + case + "_timeseries_02env.pdf")
    plt.clf()

    fig = plt.figure(1)
    ax = []
    for plot_it in range(6):
        ax.append(fig.add_subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        ax[plot_it].set_xlabel('t [hrs]')
        ax[plot_it].set_ylabel('z [m]')

    #plot0 = ax[0].pcolormesh(time, z_half, updr_buo,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot0 = ax[0].pcolormesh(time, z_half, updr_qt, rasterized=True)
    fig.colorbar(plot0, ax=ax[0], label='updr qt [g/kg]')
    #plot1 = ax[1].pcolormesh(time, z_half, updr_area * 100, cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot1 = ax[1].pcolormesh(time, z_half, updr_area * 100, rasterized=True)
    fig.colorbar(plot1, ax=ax[1], label='updr area [%]')
    #plot2 = ax[2].pcolormesh(time, z_half, updr_w,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot2 = ax[2].pcolormesh(time, z_half, updr_w, rasterized=True)
    fig.colorbar(plot2, ax=ax[2], label='updr w [m/s]')
    #plot3 = ax[3].pcolormesh(time, z_half, updr_qv,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot3 = ax[3].pcolormesh(time, z_half, updr_qv, rasterized=True)
    fig.colorbar(plot3, ax=ax[3], label='updr qv [g/kg]')
    #plot4 = ax[4].pcolormesh(time, z_half, updr_ql,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot4 = ax[4].pcolormesh(time, z_half, updr_ql, rasterized=True)
    fig.colorbar(plot4, ax=ax[4], label='updr ql [g/kg]')
    #plot5 = ax[5].pcolormesh(time, z_half, updr_qr,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    plot5 = ax[5].pcolormesh(time, z_half, updr_qr, rasterized=True)
    fig.colorbar(plot5, ax=ax[5], label='updr qr [g/kg]')

    plt.savefig(output_folder + case + "_timeseries_03updr.pdf")
    plt.clf()

    #fig = plt.figure(1)
    #fig.set_figheight(8)
    #fig.set_figwidth(30)
    #ax = []
    #for plot_it in range(3):
    #    ax.append(fig.add_subplot(1,3,plot_it+1))
    #                           #(rows, columns, number)
    #    ax[plot_it].set_xlabel('t [hrs]')
    #    ax[plot_it].set_ylabel('z [m]')

    #plot0 = ax[0].pcolormesh(time, z_half[0:200], (1 - updr_area) * 100, cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot0, ax=ax[0], label='env frac [%]')
    #plot1 = ax[1].pcolormesh(time, z_half[0:200], env_w[0:200,:],cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot1, ax=ax[1], label='env w [m/2]')
    #plot2 = ax[2].pcolormesh(time, z_half[0:200], env_ql[0:200,:],cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot2, ax=ax[2], label='env ql [g/kg]')
    #plt.savefig(output_folder + case + "_LES_comp_timeseries_02env.pdf")
    #plt.clf()

    #fig = plt.figure(1)
    #fig.set_figheight(8)
    #fig.set_figwidth(30)
    #ax = []
    #for plot_it in range(3):
    #    ax.append(fig.add_subplot(1,3,plot_it+1))
    #                           #(rows, columns, number)
    #    ax[plot_it].set_xlabel('t [hrs]')
    #    ax[plot_it].set_ylabel('z [m]')

    #plot0 = ax[0].pcolormesh(time, z_half[0:200], updr_area[0:200,:] * 100,cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot0, ax=ax[0], label='updr frac [%]')
    #plot1 = ax[1].pcolormesh(time, z_half[0:200], updr_w[0:200,:],cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot1, ax=ax[1], label='updr w [m/s]')
    #plot2 = ax[2].pcolormesh(time, z_half[0:200], updr_ql[0:200,:],cmap=discrete_cmap(8,mpl.cm.bone_r), rasterized=True)
    #fig.colorbar(plot2, ax=ax[2], label='updr ql [g/kg]')

    #plt.savefig(output_folder + case + "_LES_comp_timeseries_03updr.pdf")
    #plt.clf()

