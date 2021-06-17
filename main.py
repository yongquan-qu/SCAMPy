import argparse
import json
from pathlib import Path


def main():
    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='SCAMPy')
    parser.add_argument("namelist")
    parser.add_argument("paramlist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    file_paramlist = open(args.paramlist).read()
    paramlist = json.loads(file_paramlist)
    del file_paramlist

    # Fetch path where namelist (and presumably paramlist) is stored
    inpath = Path(args.namelist).resolve().parent

    main1d(namelist, paramlist, inpath)


def main1d(namelist, paramlist, inpath=Path.cwd()):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist, paramlist, inpath)
    Simulation.initialize(namelist)
    Simulation.run()
    print('The simulation has completed.')


if __name__ == "__main__":
    main()
