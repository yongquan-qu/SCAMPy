import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import numpy as np

import matplotlib as mpl
mpl.use('Agg')  # To allow plotting when display is off
import matplotlib.pyplot as plt
from matplotlib import ticker

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Taken from https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_mean(data, les, title, folder="plots/output/"):
    """
    Plots mean profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean = np.array(data["qt_mean"]) - np.array(data["ql_mean"])
    qv_mean_les = np.array(les["qt_mean"]) - np.array(les["ql_mean"])

    # data to plot
    x_lab  = ['QV [g/kg]', 'QL [g/kg]',      'QR [g/kg]',      'THL [K]',           'TKE [m2/s2]']
    plot_x = [qv_mean,      data["ql_mean"],  data["qr_mean"], data["thetal_mean"], data["tke_mean"]]
    plot_x_les = [qv_mean_les,      les["ql_mean"],  les["qr_mean"], les["thetali_mean"],  les["tke_mean"]]
    color  = ["navy", "darkorange"]
    label  = ["ini", "end"]

    # iteration over plots
    plots = []
    for plot_it in range(5):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        for it in range(2): #init, end
            plots[plot_it].plot(plot_x_les[plot_it][it], les["z_half"], '-', color='gray', label='les', linewidth = 4)
            plots[plot_it].plot(plot_x[plot_it][it], data["z_half"], '-', color=color[it], label=label[it], linewidth = 2)

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_drafts(data, les, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean    = np.array(data["qt_mean"])    - np.array(data["ql_mean"])
    env_qv     = np.array(data["env_qt"])     - np.array(data["env_ql"])
    updraft_qv = np.array(data["updraft_qt"]) - np.array(data["updraft_ql"])

    # data to plot
    x_lab    = ["QV [g/kg]", "QL [g/kg]",        "QR [g/kg]",        "w [m/s]",         "updraft buoyancy [cm2/s3]",  "updraft area [%]"]
    plot_upd = [qv_mean,     data["updraft_ql"], data["updraft_qr"], data["updraft_w"], data["updraft_buoyancy"],     data["updraft_area"]]
    plot_env = [env_qv,      data["env_ql"],     data["env_qr"],     data["env_w"]]
    plot_mean= [updraft_qv,  data["ql_mean"],    data["qr_mean"]]

    qv_mean_les    = np.array(les["qt_mean"])    - np.array(les["ql_mean"])
    env_qv_les     = np.array(les["env_qt"])     - np.array(les["env_ql"])
    updraft_qv_les = np.array(les["updraft_qt"]) - np.array(les["updraft_ql"])
    les_plot_upd = [qv_mean_les,     les["updraft_ql"], les["updraft_qr"], les["updraft_w"], les["updraft_b"], les["updraft_fraction"] ]
    les_plot_env = [env_qv_les,      les["env_ql"],     les["env_qr"],     les["env_w"]]
    les_plot_mean= [updraft_qv_les,  les["ql_mean"],    les["qr_mean"]]

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        #plot updrafts
        if (plot_it != 5):
            plots[plot_it].plot(les_plot_upd[plot_it][1], les["z_half"], ':', color='gray', label='les upd', linewidth = 4)
            plots[plot_it].plot(plot_upd[plot_it][1], data["z_half"], "-", color="royalblue", label="upd", linewidth = 2)
        if (plot_it == 5):
            plots[plot_it].plot(les_plot_upd[plot_it][1]* 100, les["z_half"], ':', color='gray', label='les upd', linewidth = 4)
            plots[plot_it].plot(plot_upd[plot_it][1] * 100, data["z_half"], "-", color="royalblue", label="upd", linewidth = 2)
        # plot environment
        if (plot_it < 4):
            plots[plot_it].plot(les_plot_env[plot_it][1], les["z_half"], '--', color='gray', label='les env', linewidth = 4)
            plots[plot_it].plot(plot_env[plot_it][1], data["z_half"], "-", color="darkred", label="env", linewidth = 2)
        # plot mean
        if (plot_it < 3):
            plots[plot_it].plot(les_plot_upd[plot_it][1], les["z_half"], '-', color='gray', label='les mean', linewidth = 4)
            plots[plot_it].plot(plot_mean[plot_it][1], data["z_half"], "-", color="purple", label="mean", linewidth = 2)

    plots[0].legend()
    plt.savefig(folder + title)
    plt.clf()


def plot_var_covar_mean(data, les, title, folder="plots/output/"):
    """
    Plots variance and covariance profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
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
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        plots[plot_it].plot(les[plot_var_env[plot_it]][0][1],   les["z_half"],  "-", label='_les',              c="gray",linewidth = 4)
        plots[plot_it].plot(data[plot_var_mean[plot_it]][1], data["z_half"], "-",    label=plot_var_mean[plot_it], c="crimson", linewidth = 2)
        plots[plot_it].plot(data[plot_var_env[plot_it]][1],  data["z_half"], "-",    label=plot_var_env[plot_it],  c="forestgreen", linewidth = 2)

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_var_covar_components(data, title, folder="plots/output/"):
    """
    Plots variance and covariance components profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
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
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        for var in range(5):
            plots[plot_it].plot(data[plot_var_data[plot_it][var]][1],   data["z_half"], "-", label=plot_Hvar_c[var],  c=color_c[var])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries_1D(data,  les, title, folder="plots/output/"):
    """
    Plots timeseries from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=2, markersize=6)

    # data to plot
    plot_y     = [data["updraft_cloud_cover"],  data["lwp"], data["lhf"], data["shf"], data["rd"],      data["updraft_cloud_top"], data["updraft_cloud_base"]]
    plot_les_y = [les["updraft_cloud_cover"],   les["lwp"],  les["lhf"],  les["shf"],  les["shf"],  les["updraft_cloud_top"],  les["updraft_cloud_base"]]
    y_lab  =     ['updr cl. cover',                 'lwp',       'lhf',       'shf',       'rd [m]',                      'updr CB, CT']

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        if plot_it < 4:
            plots[plot_it].set_xlabel('t [s]')
            plots[plot_it].set_ylabel(y_lab[plot_it])
            plots[plot_it].set_xlim([0, data["t"][-1]])
            plots[plot_it].grid(True)
            plots[plot_it].plot(les["t"][1:] , plot_les_y[plot_it][1:], '-', color="gray",linewidth = 4)
            plots[plot_it].plot(data["t"][1:], plot_y[plot_it][1:], '-', color="b")
            plots[plot_it].legend()
        elif plot_it == 4:
            plots[plot_it].set_xlabel('t [s]')
            plots[plot_it].set_ylabel(y_lab[plot_it])
            plots[plot_it].set_xlim([0, data["t"][-1]])
            plots[plot_it].grid(True)
            plots[plot_it].plot(data["t"][1:], plot_y[plot_it][1:], '-', color="b")
            plots[plot_it].legend()
        else:
            plots[5].set_xlabel('t [s]')
            plots[5].set_ylabel(y_lab[5])
            plots[5].set_xlim([0, data["t"][-1]])
            plots[5].grid(True)
            plots[5].plot(les["t"][1:], les["updraft_cloud_base"][1:], '-', color="gray",   label="CB_les",  linewidth = 4)
            plots[5].plot(les["t"][1:], les["updraft_cloud_top"][1:],  '-', color="gray",   label="CT_les",  linewidth = 4)
            plots[5].plot(data["t"][1:], data["updraft_cloud_base"][1:], '-', color="crimson", label="CB",  linewidth = 2)
            plots[5].plot(data["t"][1:], data["updraft_cloud_top"][1:],  '-', color="royalblue", label="CT",  linewidth = 2)

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries(data,  les, case, folder="plots/output/"):
    """
    Plots the time series of Scampy simulations

    Input:
    data   - dictionary with previousely read it data
    case   - name for the tested simulation (to be used in the plot name)
    folder - folder where to save the created plot
    """
    # customize figure parameters



    fig = plt.figure(1)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    mpl.rcParams.update({'font.size': 20})

    # read data
    z_half    = data["z_half"]
    time      = data["t"] / 60. / 60.
    mean_qv   = data["qt_mean"]    - data["ql_mean"]
    updr_qv   = data["updraft_qt"] - data["updraft_ql"]
    env_qv    = data["env_qt"]     - data["env_ql"]
    env_area  = 1. - data["updraft_area"]

    les_z_half       = les["z_half"]
    les_time         = les["t"] / 60. / 60.
    les_mean_qv      = les["qt_mean"]     - les["ql_mean"]
    updr_les_qv      = les["updraft_qt"]  - les["updraft_ql"]
    env_les_qv       = les["env_qt"]      - les["env_ql"]
    env_les_area     = 1. - les["updraft_fraction"]

    # data to plot
    mean_les_data  = [les["thetali_mean"],    les["tke_mean"],       les_mean_qv,            les["ql_mean"],            les["qr_mean"],            les["qt_mean"]]
    mean_data      = [data["thetal_mean"],    data["tke_mean"],      mean_qv,                data["ql_mean"],           data["qr_mean"],           data["qt_mean"]]
    mean_label     = ["mean thl [K]",         "mean TKE [m2/s2]",    "mean qv [g/kg]",       "mean ql [g/kg]",          "mean qr [g/kg]",          "mean qt [g/kg]"]
    mean_les_label = ["mean les thl [K]",     "mean les TKE [m2/s2]","mean les qv [g/kg]",   "mean les ql [g/kg]",      "mean les qr [g/kg]",      "mean les qt [g/kg]"]
    mean_cb        = [mpl.cm.Reds,            mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    env_les_data   = [les["env_thetali"],     env_les_area,             les["env_w"],          les["env_qt"],          les["env_ql"],             les["env_qr"]]
    env_data       = [data["env_thetal"],     env_area,                 data["env_w"],         data["env_qt"],         data["env_ql"],            data["env_qr"]]
    env_label      = ["env thl [K]",          "env area [%]",           "env w [m/s]",         "env qt [g/kg]",        "env ql [g/kg]",           "env qr [g/kg]"]
    env_les_label  = ["env les thl [K]",      "env les area [%]",       "env les w [m/s]",     "env les qt [g/kg]",    "env les ql [g/kg]",       "env les qr [g/kg]"]
    env_cb         = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds_r,         mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    updr_les_data  = [les["updraft_thetali"], les["updraft_fraction"],  les["updraft_w"],      les["updraft_qt"],      les["updraft_ql"],         les["updraft_qr"]]
    updr_data      = [data["updraft_thetal"], data["updraft_area"],     data["updraft_w"],     data["updraft_qt"],     data["updraft_ql"],        data["updraft_qr"]]
    updr_label     = ["updr thl [K]",   "     updr area [%]",           "updr w [m/s]",        "updr qv [g/kg]",       "updr ql [g/kg]",          "updr qr [g/kg"]
    updr_les_label = ["updr les thl [K]", "   updr les area [%]",       "updr les w [m/s]",    "updr les qv [g/kg]",   "updr les ql [g/kg]",      "updr les qr [g/kg"]
    updr_cb        = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    flux_data      = [data["massflux_h"],     data["diffusive_flux_h"], data["total_flux_h"],  data["massflux_qt"],    data["diffusive_flux_qt"], data["total_flux_qt"]]
    flux_label     = ["M_FL thl",             "D_FL thl ",              "tot FL thl",          "M_FL qt",              "D_FL qt",                 "tot FL qt"]
    flux_cb        = [mpl.cm.Spectral,        mpl.cm.Spectral,          mpl.cm.Spectral,       mpl.cm.Spectral_r,      mpl.cm.Spectral_r,         mpl.cm.Spectral_r]

    misc_data  = [data["eddy_viscosity"], data["eddy_diffusivity"], data["mixing_length"], data["entrainment_sc"], data["detrainment_sc"],    data["massflux"]]
    misc_label = ["eddy visc",            "eddy diff",              "mix. length",         "entr sc",              "detr sc",                 "mass flux"]
    misc_cb    = [mpl.cm.Blues,           mpl.cm.Blues,             mpl.cm.Blues,          mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    les_data_to_plot = [mean_les_data,  env_les_data,  updr_les_data]
    data_to_plot     = [mean_data,      env_data,      updr_data,      flux_data,      misc_data]
    labels           = [mean_label,     env_label,     updr_label,     flux_label,     misc_label]
    les_labels       = [mean_les_label, env_les_label, updr_les_label]
    titles           = ["01mean",       "02env",       "03updr",       "04flx",        "05misc"]
    cbs              = [mean_cb,        env_cb,        updr_cb,        flux_cb,        misc_cb]

    # iteration over plots
    for var in range(5):
        ax   = []
        plot = []
        if var<=2:
            for plot_it in range(12):
                ax.append(fig.add_subplot(4,3,plot_it+1))
                                        #(rows, columns, number)
                if plot_it+1<=6:
                    ax[plot_it].set_xlabel('t [hrs]')
                    ax[plot_it].set_ylabel('z [m]')
                    plot.append(ax[plot_it].contourf(les_time[1:], les_z_half, les_data_to_plot[var][plot_it][:,1:], cmap='RdBu_r'))
                    fig.colorbar(plot[plot_it], ax=ax[plot_it], label=les_labels[var][plot_it])
                else:
                    it = plot_it - 6

                    ax[plot_it].set_xlabel('t [hrs]')
                    ax[plot_it].set_ylabel('z [m]')
                    plot.append(ax[plot_it].contourf(time[1:], z_half, data_to_plot[var][it][:,1:], cmap='RdBu_r'))
                    fig.colorbar(plot[plot_it], ax=ax[plot_it], label=labels[var][it])
        else:
            for plot_it in range(6):
                ax.append(fig.add_subplot(2,3,plot_it+1))
                ax[plot_it].set_xlabel('t [hrs]')
                ax[plot_it].set_ylabel('z [m]')
                plot.append(ax[plot_it].contourf(time[1:], z_half, data_to_plot[var][plot_it][:,1:], cmap='RdBu_r'))
                fig.colorbar(plot[plot_it], ax=ax[plot_it], label=labels[var][plot_it])
        #plt.tight_layout()
        plt.savefig(folder + case + "_timeseries_" + titles[var] + ".pdf")
        plt.clf()
