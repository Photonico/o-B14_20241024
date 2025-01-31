#!/bin/bash
#PBS -N SCF
#PBS -P g46
#PBS -q normalsr
#PBS -o output.txt
#PBS -j oe
#PBS -l mem=500GB
#PBS -l ncpus=104
#PBS -l walltime=47:59:59
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -l software=vasp
#PBS -l storage=scratch/g46+gdata/g46

module load vasp/6.4.3

mpirun vasp_std >vasp.log
