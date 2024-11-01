#!/bin/bash
#PBS -N Convergence
#PBS -P g46
#PBS -q normal
#PBS -o output.txt
#PBS -j oe
#PBS -l mem=20GB
#PBS -l ncpus=8
#PBS -l walltime=47:59:59
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -l software=vasp
#PBS -l storage=scratch/g46+gdata/g46
#PBS -m ea
#PBS -M luke.niu@sydney.edu.au

module load vasp/6.4.3

mpirun vasp_std >vasp.log
