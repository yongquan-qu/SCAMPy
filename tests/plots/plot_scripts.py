import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import numpy as np

import matplotlib as mpl
mpl.use('Agg')  # To allow plotting when display is off
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_mean(scm_data, les_data, tmin, tmax, folder="plots/output/"):
    """
    Plots mean profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    folder   - folder where to save the created plot
    """
    t_start_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t_start_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t_end_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t_end_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    qv_mean_scm = np.array(scm_data["qt_mean"]) - np.array(scm_data["ql_mean"])
    qv_mean_les = np.array(les_data["qt_mean"]) - np.array(les_data["ql_mean"])

    x_labels  =  [r'$q_{t} mean [\mathrm{g/kg}]$',
                  r'$q_{l} mean [\mathrm{g/kg}]$',
                  r'$q_{r} mean [\mathrm{g/kg}]$',
                  r'$q_{v} mean [\mathrm{g/kg}]$',
                  r'$\theta_{l} [\mathrm{K}]$',
                  r'$TKE [\mathrm{m^2/s^2}]$',
                  'u [m/s]',
                  'v [m/s]',
                  r'$\bar{w}_{upd} [\mathrm{m/s}]$',
                  r'$\bar{b}_{upd} [\mathrm{m/s^2}]$',
                  r'$\bar{q}_{l,upd} [\mathrm{g/kg}]$',
                  r'$\bar{q}_{r,upd} [\mathrm{g/kg}]$',
                  "updraft area [%]",
                  r'$\bar{q}_{l,env} [\mathrm{g/kg}]$',
                  r'$\bar{q}_{r,env} [\mathrm{g/kg}]$']

    fig_name  =  ["qt_mean", "ql_mean", "qr_mean", "qv_mean", "thetal_mean",\
                  "TKE", "u_mean", "v_mean", "updraft_w", "updraft_buoyancy",\
                  "updraft_ql", "updraft_qr", "updraft_area", "env_ql", "env_qr"]

    plot_x_scm = [scm_data["qt_mean"], scm_data["ql_mean"], scm_data["qr_mean"],\
                  qv_mean_scm, scm_data["thetal_mean"], scm_data["tke_mean"],\
                  scm_data["u_mean"], scm_data["v_mean"], scm_data["updraft_w"],\
                  scm_data["updraft_buoyancy"], scm_data["updraft_ql"],\
                  scm_data["updraft_qr"], scm_data["updraft_area"], scm_data["env_ql"],\
                  scm_data["env_qr"]]

    plot_x_les = [les_data["qt_mean"], les_data["ql_mean"], les_data["qr_mean"],\
                  qv_mean_les, les_data["thetali_mean"], les_data["tke_mean"],\
                  les_data["u_translational_mean"], les_data["v_translational_mean"],\
                  les_data["updraft_w"], les_data["updraft_buoyancy"],\
                  les_data["updraft_ql"], les_data["updraft_qr"],\
                  les_data["updraft_fraction"], les_data["env_ql"], les_data["env_qr"]]

    color  = ["navy", "darkorange"]
    label  = ["ini", "end"]

    plots = []
    for plot_it in range(len(x_labels)):
        fig = plt.figure(fig_name[plot_it])
        plt.xlabel(x_labels[plot_it])
        plt.ylabel('height [km]')
        plt.ylim([0, scm_data["z_half"][-1]/1000.0 + (scm_data["z_half"][1]/1000.0 - scm_data["z_half"][0]/1000.0) * 0.5])
        plt.grid(True)
        plt.plot(np.nanmean(plot_x_les[plot_it][:,t_start_les:t_end_les],axis=1), les_data["z_half"],        '-', color='k', label='les', linewidth = 2)
        plt.plot(np.nanmean(plot_x_scm[plot_it][:,t_start_scm:t_end_scm],axis=1), scm_data["z_half"]/1000.0, '-', color = '#157CC7', label='scm', linewidth = 2)

        plt.legend()
        plt.tight_layout()
        plt.savefig(folder + fig_name[plot_it]+".pdf")
        plt.clf()

def plot_closures(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    scm_vars = [np.nanmean(scm_data["eddy_diffusivity"][:, t0_scm : t1_scm], axis=1),\
                np.nanmean(scm_data["mixing_length"][:,t0_scm : t1_scm] / 1e3, axis=1),\
                np.nanmean(scm_data["nh_pressure"][:,  t0_scm : t1_scm] /\
                           scm_data["updraft_area"][:, t0_scm : t1_scm], axis=1\
                          ) / scm_data["rho_half"][:],\
                np.nanmean(scm_data["turbulent_entrainment"][:, t0_scm : t1_scm], axis=1),\
                np.nanmean(scm_data["updraft_buoyancy"][:, t0_scm : t1_scm], axis=1),\
                np.nanmean(scm_data["entrainment_sc"][:, t0_scm : t1_scm], axis=1)]

    x_lab = ["eddy_diffusivity", "mixing_length [km]", "non hydro pressure [Pa]",\
             "turbulent_entrainment", "buoyancy [m/s^2]", "entr and detr [1/m]"]

    for it in range(6):
        plt.subplot(2,3,it+1)
        if it < 4:
            plt.plot(scm_vars[it], scm_data["z_half"]/1e3, "-", c="royalblue", lw=3)

        if it == 2:
            plt.plot(np.nanmean(-les_data["updraft_ddz_p_alpha"][:, t0_les : t1_les], axis=1),\
                     les_data["z_half"], '-', color='gray', label='les', lw=3)
        if it == 4:
            plt.plot(scm_vars[it], scm_data["z_half"]/1e3, "-", c="royalblue", lw=3, label="b_upd")
            plt.plot(np.nanmean(scm_data["b_mix"][:, t0_scm : t1_scm],axis=1),\
                     scm_data["z_half"]/1e3, "-", color="darkorange", label="b_mix", lw=3)
            plt.legend()
        if it == 5:

            xmax = np.min([np.max(scm_data["detrainment_sc"]), 0.05])
            if xmax == 0.0:
                xmax = np.max(scm_data["detrainment_sc"])

            plt.plot(scm_vars[it], scm_data["z_half"]/1e3, "-", c="royalblue", lw=3, label="entr")
            plt.plot(np.nanmean(scm_data["detrainment_sc"][:, t0_scm : t1_scm], axis=1),\
                     scm_data["z_half"]/1e3, "-", color="darkorange", label="detr", lw=3)
            plt.xlim([-0.0001,xmax])
            plt.legend()

        plt.xlabel(x_lab[it])
        plt.ylabel("z [km]")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_tke_components(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    x_lab =  ["tke_advection", "tke_buoy", "tke_dissipation", "tke_pressure",\
              "tke_transport","tke_shear"]

    plot_vars =  [scm_data["tke_advection"], scm_data["tke_buoy"],\
                  scm_data["tke_dissipation"], scm_data["tke_pressure"],\
                  scm_data["tke_transport"], scm_data["tke_shear"]]

    plot_x_les = [les_data["tke_prod_A"], les_data["tke_prod_B"],\
                  les_data["tke_prod_D"], les_data["tke_prod_P"],\
                  les_data["tke_prod_T"], les_data["tke_prod_S"]]

    xmax = 5*np.max(np.nanmean(scm_data["tke_entr_gain"][3:, t0_scm:t1_scm], axis=1))

    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_ylabel('z [km]')
        plots[plot_it].grid(True)
        if plot_it<6:
            # plots[plot_it].plot(np.nanmean(plot_x_les[plot_it][:, t0_les:t1_les],axis=1),\
            #                     les_data["z_half"], '-', color='gray', label='les', lw=3)
            plots[plot_it].plot(np.nanmean(plot_vars[plot_it][:, t0_scm:t1_scm],axis=1),\
                                scm_data["z_half"]/1e3, "-", color="royalblue", label='les', lw=3)
            plots[plot_it].set_xlabel(x_lab[plot_it])
            plots[plot_it].set_ylim([0, np.max(scm_data["z_half"]/1000.0)])
        else:
            plots[plot_it].plot(np.nanmean(scm_data["tke_entr_gain"][:, t0_scm:t1_scm],axis=1),\
                                scm_data["z_half"]/1e3, "-", color="royalblue", label="tke entr", lw=3)
            plots[plot_it].plot(np.nanmean(scm_data["tke_detr_loss"][:, t0_scm:t1_scm],axis=1),\
                                scm_data["z_half"]/1e3, "-", color="darkorange", label="tke detr", lw=3)
            plots[plot_it].set_xlabel('tke entr detr [1/m]')
            plots[plot_it].set_xlim([-1e-4, xmax])
            plots[plot_it].set_ylim([0, np.max(scm_data["z_half"]/1000.0)])
            plots[plot_it].legend()

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_humidities(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    scm_data["qv_mean"] = scm_data["qt_mean"]    - scm_data["ql_mean"]
    scm_data["upd_qv"]  = scm_data["updraft_qt"] - scm_data["updraft_ql"]
    scm_data["env_qv"]  = scm_data["env_qt"]     - scm_data["env_ql"]
    les_data["qv_mean"] = les_data["qt_mean"]    - les_data["ql_mean"]
    les_data["upd_qv"]  = les_data["updraft_qt"] - les_data["updraft_ql"]
    les_data["env_qv"]  = les_data["env_qt"]     - les_data["env_ql"]

    var = ["qv_mean", "upd_qv", "env_qv",\
           "ql_mean", "updraft_ql", "env_ql",\
           "qr_mean", "updraft_qr", "env_qr"]

    lab = ["mean qv [g/kg]", "updraft qv [g/kg]", "env qv [g/kg]",\
           "mean ql [g/kg]", "updraft ql [g/kg]", "env ql [g/kg]",\
           "mean qr [g/kg]", "updraft qr [g/kg]", "env qr [g/kg]"]

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    for it in range(9):
        plt.subplot(3,3,it+1)
        plt.grid(True)
        plt.xlabel(lab[it])
        plt.plot(np.nanmean(les_data[var[it]][:, t0_les:t1_les],axis=1),\
                 les_data["z_half"], '-', color='gray', label='les', lw=3)
        plt.plot(np.nanmean(scm_data[var[it]][:, t0_scm:t1_scm],axis=1),\
                 scm_data["z_half"]/1e3, "-", color="royalblue", label='les', lw=3)
        if it in [0,3,6]:
            plt.ylabel("z [km]")

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_updraft_properties(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    les_data["massflux"]  = np.multiply(les_data["updraft_fraction"], les_data["updraft_w"])

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    scm_var = ["updraft_area","updraft_w","massflux",\
               "u_mean", "thetal_mean","qt_mean"]

    les_var = ["updraft_fraction", "updraft_w", "massflux",\
               "u_translational_mean", "thetali_mean","qt_mean"]

    lab = ["updraft fraction", "updraft w [m/s]", "massflux [kg/m^2/s]",\
           "horizontal velocities [m/s]", "thetal mean [K]", "qt mean [g/kg]"]

    for it in range(6):
        plt.subplot(2,3,it+1)
        plt.grid(True)
        plt.plot(np.nanmean(les_data[les_var[it]][:, t0_les:t1_les], axis=1),\
                 les_data["z_half"], '-', color='gray', label='les', lw=3)
        plt.plot(np.nanmean(scm_data[scm_var[it]][:, t0_scm:t1_scm], axis=1),\
                 scm_data["z_half"]/1e3, "-", color="royalblue", label='scm', lw=3)
        plt.xlabel(lab[it])
        if it in [0,3]:
            plt.ylabel("z [km]")
        if it == 3:
            plt.plot(np.nanmean(les_data["v_translational_mean"][:,t0_les:t1_les],axis=1),
                     les_data["z_half"], '--', color='gray', label='v-les', lw=3)
            plt.plot(np.nanmean(scm_data["v_mean"][:, t0_scm:t1_scm], axis=1),\
                     scm_data["z_half"]/1e3, "-", color="darkorange", label='v-scm', lw=3)
            plt.legend()

    plt.savefig(folder + title)
    plt.clf()

def plot_tke_breakdown(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    # customize defaults
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    col = ["royalblue", "darkorange", "k", "darkgreen", "red", "purple"]

    scm_var = ["tke_advection","tke_buoy","tke_dissipation","tke_pressure",\
               "tke_transport","tke_shear"]

    les_var = ["tke_prod_A", "tke_prod_B", "tke_prod_D", "tke_prod_P",\
               "tke_prod_T", "tke_prod_S"]

    plt.subplot(121)
    for it in range(6):
        plt.plot(np.nanmean(scm_data[scm_var[it]][:, t0_scm:t1_scm], axis=1),\
                 scm_data["z_half"]/1e3, "-", color=col[it],  label=scm_var[it],\
                 lw=3)
    plt.ylim([0, np.max(scm_data["z_half"]/1e3)])
    plt.xlabel('tke componenets scm')
    plt.ylabel('height [km]')
    plt.legend()

    plt.subplot(122)
    for it in range(6):
        plt.plot(np.nanmean(les_data[les_var[it]][:, t0_les:t1_les], axis=1),\
                 les_data["z_half"], "-", color=col[it],  label=les_var[it],\
                 lw=3)
    plt.ylim([0, np.max(les_data["z_half"])])
    plt.xlabel('tke componenets les')
    plt.legend()

    plt.savefig(folder + title)
    plt.clf()

def plot_var_covar_mean(scm_data, les_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots variance and covariance profiles from Scampy

    Input:
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    title    - name for the created plot
    folder   - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t0_les = int(np.where(np.array(les_data["t"]) > tmin)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])
    t1_les = int(np.where(np.array(tmax<= les_data["t"]))[0][0])

    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    x_lab         = ["Hvar",       "QTvar",       "HQTcov"]
    plot_var_mean = ["Hvar_mean",  "QTvar_mean",  "HQTcov_mean"]
    plot_var_env  = ["env_Hvar",   "env_QTvar",   "env_HQTcov"]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('height [km]')
        plots[plot_it].set_ylim([0,\
                                 scm_data["z_half"][-1]/1000.0 +\
                                 (scm_data["z_half"][1]/1000.0 - scm_data["z_half"][0]/1000.0) * 0.5\
                                ])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        plots[plot_it].plot(np.nanmean(les_data[plot_var_env[plot_it]][:, t0_les:t1_les], axis=1),\
                            les_data["z_half"], "-", label= 'les', c="gray", lw=4)
        plots[plot_it].plot(np.nanmean(scm_data[plot_var_mean[plot_it]][:,t0_scm:t1_scm], axis=1),\
                            scm_data["z_half"]/1e3, "-", label=plot_var_mean[plot_it], c="crimson", lw=3)
        plots[plot_it].plot(np.nanmean(scm_data[plot_var_env[plot_it]][:, t0_scm:t1_scm], axis=1),\
                            scm_data["z_half"]/1e3, "-", label=plot_var_env[plot_it],  c="forestgreen", lw=3)

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_var_covar_components(scm_data, tmin, tmax, title, folder="plots/output/"):
    """
    Plots variance and covariance components profiles from Scampy

    Input:
    scm_data   - scm stats file
    tmin   - lower bound for time mean
    tmax   - upper bound for time mean
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    t0_scm = int(np.where(np.array(scm_data["t"]) > tmin*3600.0)[0][0])
    t1_scm = int(np.where(np.array(tmax*3600.0<= scm_data["t"]))[0][0])

    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    plot_Hvar_c   = ["Hvar_dissipation",   "Hvar_entr_gain",   "Hvar_detr_loss",   "Hvar_shear",   "Hvar_rain"]
    plot_QTvar_c  = ["QTvar_dissipation",  "QTvar_entr_gain",  "QTvar_detr_loss",  "QTvar_shear",  "QTvar_rain"]
    plot_HQTcov_c = ["HQTcov_dissipation", "HQTcov_entr_gain", "HQTcov_detr_loss", "HQTcov_shear", "HQTcov_rain"]
    color_c       = ['darkgreen',              'purple',             'purple',           'darkorange',       'royalblue']

    x_lab         = ["Hvar",      "QTvar",      "HQTcov"]
    plot_var_data = [plot_Hvar_c, plot_QTvar_c, plot_HQTcov_c]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('height [km]')
        plots[plot_it].set_ylim([0,\
                                 scm_data["z_half"][-1]/1e3 +\
                                 (scm_data["z_half"][1]/1e3 - scm_data["z_half"][0]/1e3) * 0.5\
                                ])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        for var in range(5):
            plots[plot_it].plot(np.nanmean(scm_data[plot_var_data[plot_it][var]][:, t0_scm:t1_scm], axis=1),\
                                scm_data["z_half"]/1e3, "-", label=plot_Hvar_c[var], c=color_c[var])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()

def plot_main_timeseries(scm_srs, les_srs, scm_data, les_data, title , folder="plots/output/"):
    """
    Plots the time series of Scampy simulations

    Input:
    scm_srs  - scm timeseries file
    les_srs  - les timeseries file
    scm_data - scm stats file
    les_data - les stats file
    tmin     - lower bound for time mean
    tmax     - upper bound for time mean
    folder   - folder where to save the created plot
    """
    # customize figure parameters
    # read data
    scm_z_half = scm_data["z_half"]/1000.0
    scm_time   = scm_data["t"] /3600.0
    les_z_half = les_data["z_half"]
    les_time   = les_data["t"]

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    cmap = "RdBu_r"
    cb_min = [ , ]
    cb_max = [ , ]

    les_var = ["ql_mean", "updraft_w"]
    les_tit = ["les ql mean", "les upd w"]
    for it in range(2):
        plt.subplot(3,2,it+1)
        levels = np.linspace(cb_min[plot_it], cb_max[plot_it], 11)
        cntrf = plt.contourf(les_time, les_z_half, les_data[les_var[it]],\
                             cmap=cmap, levels=levels, vmin=cb_min[plot_it], vmax=cb_max[plot_it])
        cbar = plt.colorbar(cntrf)
        cbar.set_label(labels[plot_it])
        plt.ylim([0, np.max(scm_z_half)])
        plt.ylabel('height [km]')
        plt.title(les_tit[it])

    scm_var = ["ql_mean", "updraft_w"]
    scm_tit = ["scm ql mean", "scm upd w"]
    for it in range(2):
        plt.subplot(3,2,it+3)
        plt.contourf(scm_time, scm_z_half, scm_data[scm_var[it]], cmap='RdBu_r')
        plt.ylim([0, np.max(scm_z_half)])
        plt.xlabel('time [h]')
        plt.ylabel('height [km]')
        plt.colorbar()
        plt.title(scm_tit[it])

    # TODO add rwp
    for it in range(2):
        plt.subplot(3,2,it+5)
        plt.plot(les_srs["t"][1:],        les_srs["lwp_mean"][1:], '-', c="gray", lw=3)
        plt.plot(scm_srs["t"][1:]/3600.0, scm_srs["lwp_mean"][1:], '-', c="royalblue", lw=3)
        plt.xlim([0, scm_srs["t"][-1]/3600.0])
        plt.xlabel('time [h]')
        plt.ylabel("lwp ")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()
    plt.close()

def plot_timeseries_1D(data,  les, folder="plots/output/"):
    """
    Plots timeseries from Scampy

    Input:
    data   - scm stats file
    les    - les stats file
    tmin   - lower bound for time mean
    tmax   - upper bound for time mean
    folder - folder where to save the created plot
    """
    # customize defaults

    # data to plot
    plot_y     = [data["cloud_cover_mean"],  data["lwp_mean"], data["lhf"], data["shf"], data["rd"],             data["cloud_top_mean"], data["cloud_base_mean"]]
    plot_les_y = [les["cloud_cover_mean"],   les["lwp_mean"],  les["lhf"],  les["shf"],  les["shf"],             les["cloud_top_mean"],  les["cloud_base_mean"]]
    y_lab  =     ['cloud cover',                 'lwp',       'lhf',       'shf',       'rd [m]',                      'CB, CT [km]']
    fig_name  =  ['cloud_cover', 'liquid_water_path',  'latent_heat_flux',  'sensible_heat_flux',       'plume_separation_radius',   'cloud_base_top']

    # iteration over plots
    plots = []
    for plot_it in range(6):
        fig = plt.figure(1)
        if plot_it < 4:
            plt.xlabel('time [h]')
            plt.ylabel(y_lab[plot_it])
            plt.xlim([0, data["t"][-1]/3600.0])
            plt.grid(True)
            plt.plot(les["t"][1:] , plot_les_y[plot_it][1:], '-', color="gray",linewidth = 4)
            plt.plot(data["t"][1:]/3600.0, plot_y[plot_it][1:], '-', color="b")
            plt.legend()
        elif plot_it == 4:
            plt.xlabel('time [h]')
            plt.ylabel(y_lab[plot_it])
            plt.xlim([0, data["t"][-1]/3600.0])
            plt.grid(True)
            plt.plot(data["t"][1:]/3600.0, plot_y[plot_it][1:], '-', color="b")
            plt.legend()
            plt.tight_layout()
        else:
            plt.xlabel('time [h]')
            plt.ylabel(y_lab[5])
            plt.xlim([0, data["t"][-1]/3600.0])
            plt.grid(True)
            plt.plot(les["t"][1:], les["cloud_base_mean"][1:], '-', color="gray",   label="CB_les",  linewidth = 4)
            plt.plot(les["t"][1:], les["cloud_top_mean"][1:],  '-', color="gray",   label="CT_les",  linewidth = 4)
            plt.plot(data["t"][1:]/3600.0, data["cloud_base_mean"][1:], '-', color="crimson", label="CB",  linewidth = 2)
            plt.plot(data["t"][1:]/3600.0, data["cloud_top_mean"][1:],  '-', color="royalblue", label="CT",  linewidth = 2)

        plt.savefig(folder + fig_name[plot_it]+".pdf")
        plt.clf()

def plot_contour_timeseries(data,  les, folder="plots/output/"):
    """
    Plots the time series of Scampy simulations

    Input:
    data   - scm stats file
    les    - les stats file
    tmin   - lower bound for time mean
    tmax   - upper bound for time mean
    folder - folder where to save the created plot
    """
    # customize figure parameters
    # read data
    z_half    = data["z_half"]/1000.0
    time      = data["t"] /3600.0
    data["qv_mean"]  = data["qt_mean"]    - data["ql_mean"]
    data["upd_qv"] = data["updraft_qt"] - data["updraft_ql"]
    data["env_qv"]  = data["env_qt"]     - data["env_ql"]

    les_z_half     = les["z_half"]
    les_time       = les["t"]
    les["qv_mean"] = les["qt_mean"]     - les["ql_mean"]
    les["upd_qv"]  = les["updraft_qt"]  - les["updraft_ql"]
    les["env_qv"]  = les["env_qt"]      - les["env_ql"]

    # data to plot "qt_mean",
    les_vars  = ["thetali_mean", "tke_mean", "qv_mean", "ql_mean", "qr_mean", "qt_mean", "env_thetali", "env_w", "env_qt", "env_ql","env_qr",
                 "updraft_thetali", "updraft_fraction", "updraft_buoyancy", "updraft_w", "updraft_qt", "updraft_ql","updraft_qr",
                 "massflux_h", "diffusive_flux_h", "total_flux_h", "massflux_qt", "diffusive_flux_qt", "total_flux_qt", "u_translational_mean", "v_translational_mean"]

    scm_vars  = ["thetal_mean", "tke_mean", "qv_mean", "ql_mean", "qr_mean", "qt_mean", "env_thetal", "env_w", "env_qt", "env_ql", "env_qr",
                 "updraft_thetal", "updraft_area", "updraft_buoyancy", "updraft_w", "updraft_qt", "updraft_ql", "updraft_qr",
                 "massflux_h", "diffusive_flux_h", "total_flux_h", "massflux_qt", "diffusive_flux_qt", "total_flux_qt", "u_mean", "v_mean"]

    labels    = ["mean thl [K]", "mean TKE [m2/s2]", "mean qv [g/kg]", "mean ql [g/kg]", "mean qr [g/kg]", "mean qt [g/kg]", "env thl [K]", "env w [m/s]",
                 "env qt [g/kg]", "env ql [g/kg]", "env qr [g/kg]", "updr thl [K]", "updr area [%]", "updr buoyancy [m/s^2]", "updr w [m/s]",
                 "updr qt [g/kg]", "updr ql [g/kg]", "updr qr [g/kg", "massflux_h [kg*K/ms^2]", "diffusive_flux_h [kg*K/ms^2]", "total_flux_h [kg*K/ms^2]",
                 "massflux_qt [g*/ms^2]", "diffusive_flux_qt [g*/ms^2]", "total_flux_qt [g*/ms^2]",  "u [m/s]", "v [m/s]"]

    fig_name =  ["contour_thl_mean", "contour_TKE_mean", "contour_qv_mean", "contour_ql_mean", "contour_qr_mean", "contour_qt_mean", "contour_env_thl", "contour_env_w",
                 "contour_env_qt", "contour_env_ql", "contour_env_qr", "contour_upd_thl", "contour_upd_area", "contour_upd_buoyancy", "contour_upd_w", "contour_upd_qt",
                 "contour_upd_ql", "contour_upd_qr", "contour_massflux_h", "contour_diffusive_flux_h", "contour_total_flux_h", "contour_massflux_qt", "contour_diffusive_flux_qt",
                 "contour_total_flux_qt","contour_u_mean", "contour_v_mean"]

    # iteration over plots
    plots = []
    for plot_it in range(len(labels)):
        fig = plt.figure(fig_name[plot_it])
        # there is a bag in the initial condition for env thetal in scampy, its starts with zeros, this quick fix should be removed after the bag is fixed
        if scm_vars[plot_it]=="env_thetal":
            data[scm_vars[plot_it]][:,0] = data[scm_vars[plot_it]][:,1]
        scm_field = np.multiply(data[scm_vars[plot_it]],1.0)
        les_field = np.multiply(les[les_vars[plot_it]],1.0)
        a_scm = np.multiply(data['updraft_area'],1.0)
        a_les = np.multiply(les['updraft_fraction'],1.0)
        if ("updraft" in scm_vars[plot_it]):
            scm_field[np.where(a_scm==0.0)] = np.nan
            scm_field[np.where(np.isnan(a_scm))] = np.nan

        if ("updraft" in les_vars[plot_it]):
            les_field[np.where(a_les==0.0)] = np.nan
            les_field[np.where(np.isnan(a_les))] = np.nan
        plt.subplot(211)
        plt.ylabel('height [km]')
        plt.contourf(les_time, les_z_half, les_field, cmap='RdBu_r')
        plt.colorbar()
        plt.ylim([0,np.max(data["z_half"]/1000.0)])
        plt.subplot(212)
        plt.xlabel('time [h]')
        plt.ylabel('height [km]')
        plt.contourf(time, z_half, scm_field, cmap='RdBu_r')
        plt.colorbar()
        plt.ylim([0,np.max(data["z_half"]/1000.0)])
        plt.tight_layout()
        plt.savefig(folder + fig_name[plot_it]+".pdf")
        plt.clf()
        plt.close()
