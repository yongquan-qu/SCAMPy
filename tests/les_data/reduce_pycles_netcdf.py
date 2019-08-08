import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse

# command line:              input                                 output
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc  /Users/yaircohen/Desktop/Bomex.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/clima_master/TRMM_LBA_TL/standard2/Stats.TRMM_LBA.nc /Users/yaircohen/Desktop/TRMM_LBA.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/simualtions_stats/Output.GATE_III.Tracers_no_evap/Stats.GATE_III.nc  /Users/yaircohen/Desktop/GATE_III.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/stats/staRico_TL/Stats.Rico.nc  /Users/yaircohen/Desktop/Rico.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/clima_master/DyCOMS_RF01/stats/Stats.DYCOMS_RF01.nc  /Users/yaircohen/Desktop/DYCOMS_RF01.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/clima_master/Soares/stats/Stats.Soares.nc  /Users/yaircohen/Desktop/Soares.nc
# python reduce_pycles_netcdf.py /Users/yaircohen/Documents/PyCLES_out/GABLS/Stats.Gabls.nc   /Users/yaircohen/Documents/codes/scampy/tests/les_data/Gabls.nc
def main():
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("fullfilename")
    parser.add_argument("fname")
    args = parser.parse_args()
    fullfilename = args.fullfilename
    fname = args.fname

    data = nc.Dataset(fullfilename, 'r')

    env_w_ = data.groups['profiles'].variables['env_w']
    temperature_mean_ = data.groups['profiles'].variables['temperature_mean']
    u_mean_ = data.groups['profiles'].variables['u_mean']
    v_mean_ = data.groups['profiles'].variables['v_mean']
    tke_mean_ = data.groups['profiles'].variables['tke_mean']
    v_translational_mean_ = data.groups['profiles'].variables['v_translational_mean']
    u_translational_mean_ = data.groups['profiles'].variables['u_translational_mean']
    updraft_b_ = data.groups['profiles'].variables['updraft_b']
    updraft_fraction_ = data.groups['profiles'].variables['updraft_fraction']


    # try the TKE diagnostics outputs
    try:
        tke_prod_A_ = data.groups['profiles'].variables['tke_prod_A']
        tke_prod_B_ = data.groups['profiles'].variables['tke_prod_B']
        tke_prod_D_ = data.groups['profiles'].variables['tke_prod_D']
        tke_prod_P_ = data.groups['profiles'].variables['tke_prod_P']
        tke_prod_T_ = data.groups['profiles'].variables['tke_prod_T']
        tke_prod_S_ = data.groups['profiles'].variables['tke_prod_S']
        tke_nd_mean_ = data.groups['profiles'].variables['tke_nd_mean']
    except:
        tke_prod_A_ = np.zeros_like(env_w_)
        tke_prod_B_ = np.zeros_like(env_w_)
        tke_prod_D_ = np.zeros_like(env_w_)
        tke_prod_P_ = np.zeros_like(env_w_)
        tke_prod_T_ = np.zeros_like(env_w_)
        tke_prod_S_ = np.zeros_like(env_w_)
        tke_nd_mean_ = np.zeros_like(env_w_)
    try:
        qr_mean_ = data.groups['profiles'].variables['qr_mean']
        env_qr_ = data.groups['profiles'].variables['env_qr']
        updraft_qr_ = data.groups['profiles'].variables['updraft_qr']
    except:
        qr_mean_ = np.zeros_like(env_w_)
        env_qr_ = np.zeros_like(env_w_)
        updraft_qr_ = np.zeros_like(env_w_)
    try:
        qt_mean_ = data.groups['profiles'].variables['qt_mean']
        qt_mean2_ = data.groups['profiles'].variables['qt_mean2']
        env_qt_ = data.groups['profiles'].variables['env_qt']
        env_qt2_ = data.groups['profiles'].variables['env_qt2']
        updraft_qt_ = data.groups['profiles'].variables['updraft_qt']
    except:
        qt_mean_ = np.zeros_like(env_w_)
        qt_mean2_ = np.zeros_like(env_w_)
        env_qt_ = np.zeros_like(env_w_)
        env_qt2_ = np.zeros_like(env_w_)
        updraft_qt_ = np.zeros_like(env_w_)
    try:
        ql_mean_ = data.groups['profiles'].variables['ql_mean']
        env_ql_ = data.groups['profiles'].variables['env_ql']
        updraft_ql_ = data.groups['profiles'].variables['updraft_ql']
    except:
        ql_mean_ = np.zeros_like(env_w_)
        env_ql_ = np.zeros_like(env_w_)
        updraft_ql_ = np.zeros_like(env_w_)
    try:
        env_qt_thetali_ = data.groups['profiles'].variables['env_qt_thetali']
    except:
        env_qt_thetali_ = np.zeros_like(env_w_)

    try:
        thetali_mean_ = data.groups['profiles'].variables['thetali_mean']
        thetali_mean2_ = data.groups['profiles'].variables['thetali_mean2']
    except:
        thetali_mean_ = data.groups['profiles'].variables['theta_mean']
        thetali_mean2_ = data.groups['profiles'].variables['theta_mean2']

    env_thetali_ = data.groups['profiles'].variables['env_thetali']
    env_thetali2_ = data.groups['profiles'].variables['env_thetali2']
    updraft_thetali_ = data.groups['profiles'].variables['updraft_thetali']
    updraft_w_ = data.groups['profiles'].variables['updraft_w']

    friction_velocity_mean_ = data.groups['timeseries'].variables['friction_velocity_mean']
    shf_surface_mean_ = data.groups['timeseries'].variables['shf_surface_mean']
    lhf_surface_mean_ = data.groups['timeseries'].variables['lhf_surface_mean']
    try:
        cloud_fraction_ = data.groups['timeseries'].variables['cloud_fraction']
        cloud_base_ = data.groups['timeseries'].variables['cloud_base']
        cloud_top_ = data.groups['timeseries'].variables['cloud_top']
    except:
        cloud_fraction_ = np.zeros_like(lhf_surface_mean_)
        cloud_base_ = np.zeros_like(lhf_surface_mean_)
        cloud_top_ = np.zeros_like(lhf_surface_mean_)
    try:
        lwp_ = data.groups['timeseries'].variables['lwp']
    except:
    	lwp_ = np.zeros_like(lhf_surface_mean_)
    # thetali_srf_int_ = data.groups['timeseries'].variables['thetali_srf_int']

    z_ = data.groups['profiles'].variables['z']
    t_ = data.groups['profiles'].variables['t']

    output = nc.Dataset(fname, "w", format="NETCDF4")
    output.createDimension('z', len(z_))
    output.createDimension('t', len(t_))
    output.createDimension('dim', None)
    output.createGroup("profiles")
    output.createGroup("timeseries")

    t = output.createVariable('t', 'f4', 't')
    z = output.createVariable('z', 'f4', 'z')

    profiles_grp = output.groups["profiles"]

    temperature_mean = profiles_grp.createVariable('temperature_mean','f4',('t','z'))
    thetali_mean = profiles_grp.createVariable('thetali_mean','f4',('t','z'))
    qt_mean = profiles_grp.createVariable('qt_mean','f4',('t','z'))
    ql_mean = profiles_grp.createVariable('ql_mean','f4',('t','z'))
    u_mean = profiles_grp.createVariable('u_mean','f4',('t','z'))
    v_mean = profiles_grp.createVariable('v_mean','f4',('t','z'))
    tke_mean = profiles_grp.createVariable('tke_mean','f4',('t','z'))
    v_translational_mean = profiles_grp.createVariable('v_translational_mean','f4',('t','z'))
    u_translational_mean = profiles_grp.createVariable('u_translational_mean','f4',('t','z'))
    updraft_b = profiles_grp.createVariable('updraft_b','f4',('t','z'))
    updraft_fraction = profiles_grp.createVariable('updraft_fraction','f4',('t','z'))
    env_thetali = profiles_grp.createVariable('env_thetali','f4',('t','z'))
    updraft_thetali = profiles_grp.createVariable('updraft_thetali','f4',('t','z'))
    env_qt = profiles_grp.createVariable('env_qt','f4',('t','z'))
    updraft_qt = profiles_grp.createVariable('updraft_qt','f4',('t','z'))
    env_ql = profiles_grp.createVariable('env_ql','f4',('t','z'))
    updraft_ql = profiles_grp.createVariable('updraft_ql','f4',('t','z'))
    qr_mean = profiles_grp.createVariable('qr_mean','f4',('t','z'))
    env_qr = profiles_grp.createVariable('env_qr','f4',('t','z'))
    updraft_qr = profiles_grp.createVariable('updraft_qr','f4',('t','z'))
    updraft_w = profiles_grp.createVariable('updraft_w','f4',('t','z'))
    env_w = profiles_grp.createVariable('env_w','f4',('t','z'))
    thetali_mean2 = profiles_grp.createVariable('thetali_mean2','f4',('t','z'))
    qt_mean2 = profiles_grp.createVariable('qt_mean2','f4',('t','z'))
    env_thetali2 = profiles_grp.createVariable('env_thetali2','f4',('t','z'))
    env_qt2 = profiles_grp.createVariable('env_qt2','f4',('t','z'))
    env_qt_thetali = profiles_grp.createVariable('env_qt_thetali','f4',('t','z'))
    tke_prod_A = profiles_grp.createVariable('tke_prod_A','f4',('t','z'))
    tke_prod_B = profiles_grp.createVariable('tke_prod_B','f4',('t','z'))
    tke_prod_D = profiles_grp.createVariable('tke_prod_D','f4',('t','z'))
    tke_prod_P = profiles_grp.createVariable('tke_prod_P','f4',('t','z'))
    tke_prod_T = profiles_grp.createVariable('tke_prod_T','f4',('t','z'))
    tke_prod_S = profiles_grp.createVariable('tke_prod_S','f4',('t','z'))
    tke_nd_mean = profiles_grp.createVariable('tke_nd_mean','f4',('t','z'))

    timeseries_grp = output.groups['timeseries']
    cloud_fraction = timeseries_grp.createVariable('cloud_fraction','f4','t')
    cloud_base = timeseries_grp.createVariable('cloud_base','f4','t')
    cloud_top = timeseries_grp.createVariable('cloud_top','f4','t')
    friction_velocity_mean = timeseries_grp.createVariable('friction_velocity_mean','f4','t')
    shf_surface_mean = timeseries_grp.createVariable('shf_surface_mean','f4','t')
    lhf_surface_mean = timeseries_grp.createVariable('lhf_surface_mean','f4','t')
    lwp = timeseries_grp.createVariable('lwp','f4','t')
    # thetali_srf_int = timeseries_grp.createVariable('thetali_srf_int','f4','t')

    temperature_mean[:,:] = temperature_mean_[:,:]
    thetali_mean[:,:] = thetali_mean_[:,:]
    qt_mean[:,:] = qt_mean_[:,:]
    ql_mean[:,:] = ql_mean_[:,:]
    u_mean[:,:] = u_mean_[:,:]
    v_mean[:,:] = v_mean_[:,:]
    tke_mean[:,:] = tke_mean_[:,:]
    v_translational_mean[:,:] = v_translational_mean_[:,:]
    u_translational_mean[:,:] = u_translational_mean_[:,:]
    updraft_b[:,:] = updraft_b_[:,:]
    updraft_fraction[:,:] = updraft_fraction_[:,:]
    env_thetali[:,:] = env_thetali_[:,:]
    updraft_thetali[:,:] = updraft_thetali_[:,:]
    env_qt[:,:] = env_qt_[:,:]
    updraft_qt[:,:] = updraft_qt_[:,:]
    env_ql[:,:] = env_ql_[:,:]
    updraft_ql[:,:] = updraft_ql_[:,:]
    qr_mean[:,:] = qr_mean_[:,:]
    env_qr[:,:] = env_qr_[:,:]
    updraft_qr[:,:] = updraft_qr_[:,:]
    updraft_w[:,:] = updraft_w_[:,:]
    env_w[:,:] = env_w_[:,:]
    thetali_mean2[:,:] = thetali_mean2_[:,:]
    qt_mean2[:,:] = qt_mean2_[:,:]
    env_thetali2[:,:] = env_thetali2_[:,:]
    env_qt2[:,:] = env_qt2_[:,:]
    env_qt_thetali[:,:] = env_qt_thetali_[:,:]
    tke_prod_A[:,:] = tke_prod_A_[:,:]
    tke_prod_B[:,:] = tke_prod_B_[:,:]
    tke_prod_D[:,:] = tke_prod_D_[:,:]
    tke_prod_P[:,:] = tke_prod_P_[:,:]
    tke_prod_T[:,:] = tke_prod_T_[:,:]
    tke_prod_S[:,:] = tke_prod_S_[:,:]
    tke_nd_mean[:,:] = tke_nd_mean_[:,:]

    cloud_fraction[:] = cloud_fraction_[:]
    cloud_base[:] = cloud_base_[:]
    cloud_top[:] = cloud_top_[:]
    friction_velocity_mean[:] = friction_velocity_mean_[:]
    shf_surface_mean[:] = shf_surface_mean_[:]
    lhf_surface_mean[:] = lhf_surface_mean_[:]
    lwp[:] = lwp_[:]

    z[:] = z_[:]
    t[:] = t_[:]
    output.close()

if __name__ == '__main__':
    main()