#!/bin/bash
#
#SBATCH --job-name=pinball
#SBATCH --output=log.out
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=168:00:00
#

module load gcc openmpi vtk/latest felicia/latest mtc/tsv eigen/latest hdf5/latest cmake/latest git/latest petsc/latest mtc/latest
module load cimlibxx/master
module load cimlibxx/drl/pbo

export PYTHONPATH=/home/mbailly1/.local/lib/python3.6/site-packages:$PYTHONPATH

pbo airfoil.json
