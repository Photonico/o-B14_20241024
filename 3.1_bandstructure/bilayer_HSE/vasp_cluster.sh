#!/bin/csh
#PBS -N BS_HSE06
#PBS -q cmt
#PBS -j oe
#PBS -l select=1:ncpus=168:mpiprocs=168:mem=500GB
#PBS -l walltime=48:00:00
#PBS -m a
#PBS -M luke.niu@sydney.edu.au

cd "$PBS_O_WORKDIR"

module load pbspro
module load oneapi-2024.2/compiler-rt32/latest
module load oneapi-2024.2/mkl/latest
module load oneapi-2024.2/mpi/latest    
module load hdf/5/1.14.1-2_intel2021

set VASP=/cmt2/ocon2505/VASP/vasp.6.5.0/bin/vasp_std
set BIN=/cmt2/ocon2505/VASP/vasp.6.5.0/bin/vasp_std

mpirun -np 168 $VASP > vasp_cluster.out
