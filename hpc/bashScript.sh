#!/bin/bash

# job name, will show up in squeue output
#SBATCH --job-name=cp-dyn

# mail to which notifications will be sent 
#SBATCH --mail-user=nguyed99@zedat.fu-berlin.de

# type of email notification - BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL,END

# ensure that all cores are on one machine
#SBATCH --nodes=1

# number of cores
#SBATCH --ntasks=1

# number of CPUs per task
#SBATCH --cpus-per-task=3

# memory per CPU in MB (see also --mem)
#SBATCH --mem-per-cpu=2048

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=/scratch/nguyed99/qcp-1d-julia/logging/cp_dyn_%A_%a.out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=/scratch/nguyed99/qcp-1d-julia/logging/cp_dyn_%A_%a.err

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=2-00:00:00

# job arrays
#SBATCH --array=0-2

# select partition
#SBATCH --partition=main

# set specific queue/nodes
# SBATCH --reservation=bqa


# load Julia module
module load julia/1.10.2

# set number of CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


# simulation parameter
N=10
OMEGAS=(2.0 6.0 10.0)
OMEGA=${OMEGAS[$SLURM_ARRAY_TASK_ID]}

BONDDIM=25
KRAUSDIM=15
dt=0.1
nt=50

# paths and file names
timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
ENV_PATH="/scratch/nguyed99/qcp-1d-julia/"


# store job info in output file, if you want...
scontrol show job $SLURM_JOBID
echo "slurm task ID = $SLURM_ARRAY_TASK_ID"
echo $N $OMEGA $BONDDIM $KRAUSDIM $dt $nt

# launch Julia script
ENV_PATH="/scratch/nguyed99/qcp-1d-julia/"
export JULIA_PROJECT=$ENV_PATH
julia dynamics_hpc.jl --N $N --OMEGA $OMEGA --BONDDIM $BONDDIM --KRAUSDIM $KRAUSDIM --dt $dt --nt $nt --JOBID $SLURM_ARRAY_JOB_ID