import argparse
import json
import pprint
from sys import exit
import uuid
import ast
import copy

# See Table 1 of Tan et al, 2018
#paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] ==> c_k (scaling constant for eddy diffusivity/viscosity
#paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] == > c_e (scaling constant for tke dissipation)
#paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] ==> alpha_b (scaling constant for virtual mass term)
#paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] ==> alpha_d (scaling constant for drag term)
# paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] ==> r_d (horizontal length scale of plume spacing)

# Parameters below can be used to multiply any entrainment rate for quick tuning/experimentation
# (NOTE: these are not c_epsilon, c_delta,0 defined in Tan et al 2018)
# paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.1
# paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 1.0

#NB: except for Bomex and life_cycle_Tan2018 cases, the parameters listed have not been thoroughly tuned/tested
# and should be regarded as placeholders only. Optimal parameters may also depend on namelist options, such as
# entrainment/detrainment rate formulation, diagnostic vs. prognostic updrafts, and vertical resolution

def main():

    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')

    args = parser.parse_args()

    case_name = args.case_name

    paramlist_defaults = {}
    paramlist_defaults['meta'] = {}

    paramlist_defaults['turbulence'] = {}
    paramlist_defaults['turbulence']['prandtl_number'] = 1.0
    paramlist_defaults['turbulence']['Ri_bulk_crit'] = 0.2

    paramlist_defaults['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.16
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.35
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 9.9
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.03
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 3.0
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const'] = 0.5
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0/3.0
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['aspect_ratio'] = 0.25

    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_coeff1'] = 1.0/3.0
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_coeff2'] = 0.0
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_coeff3'] = 0.75
    paramlist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_coeff3'] = 1.0

    if case_name == 'Soares':
        paramlist = Soares(paramlist_defaults)
    elif case_name == 'Bomex':
        paramlist = Bomex(paramlist_defaults)
    elif case_name == 'life_cycle_Tan2018':
        paramlist = life_cycle_Tan2018(paramlist_defaults)
    elif case_name == 'Rico':
        paramlist = Rico(paramlist_defaults)
    elif case_name == 'TRMM_LBA':
        paramlist = TRMM_LBA(paramlist_defaults)
    elif case_name == 'ARM_SGP':
        paramlist = ARM_SGP(paramlist_defaults)
    elif case_name == 'GATE_III':
        paramlist = GATE_III(paramlist_defaults)
    elif case_name == 'DYCOMS_RF01':
        paramlist = DYCOMS_RF01(paramlist_defaults)
    elif case_name == 'GABLS':
        paramlist = GABLS(paramlist_defaults)
    elif case_name == 'SP':
        paramlist = SP(paramlist_defaults)
    else:
        print('Not a valid case name')
        exit()

    write_file(paramlist)

def Soares(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'Soares'

    return paramlist

def Bomex(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'Bomex'

    return  paramlist

def life_cycle_Tan2018(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'life_cycle_Tan2018'

    return  paramlist

def Rico(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'Rico'

    return  paramlist

def TRMM_LBA(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'TRMM_LBA'

    return  paramlist

def ARM_SGP(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'ARM_SGP'

    return  paramlist

def GATE_III(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'GATE_III'

    return  paramlist

def DYCOMS_RF01(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'DYCOMS_RF01'

    paramlist['turbulence']['prandtl_number'] = 0.8

    return  paramlist

def GABLS(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'GABLS'

    paramlist['turbulence']['prandtl_number'] = 0.74

    return  paramlist

# Not fully implemented yet - Ignacio
def SP(paramlist_defaults):

    paramlist = copy.deepcopy(paramlist_defaults)

    paramlist['meta']['casename'] = 'SP'

    return  paramlist

def write_file(paramlist):

    fh = open('paramlist_'+paramlist['meta']['casename']+ '.in', 'w')
    #pprint.pprint(paramlist)
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
