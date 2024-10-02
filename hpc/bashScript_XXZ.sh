#!/bin/bash

# job name, will show up in squeue output
#SBATCH --job-name=xxz-dyn

# mail to which notifications will be sent 
#SBATCH --mail-user=nguyed99@zedat.fu-berlin.de

# type of email notification - BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL,END

# ensure that all cores are on one machine
#SBATCH --nodes=1

# number of cores
#SBATCH --ntasks=1

# number of CPUs per task
#SBATCH --cpus-per-task=16

# memory per CPU in MB (see also --mem)
#SBATCH --mem-per-cpu=4096

# file to which standard output will be written (%A --> jobID, %a --> arrayID)
#SBATCH --output=/scratch/nguyed99/qcp-1d-julia/logging/xxz_dyn_%A_%a.out

# file to which standard errors will be written (%A --> jobID, %a --> arrayID)
#SBATCH --error=/scratch/nguyed99/qcp-1d-julia/logging/xxz_dyn_%A_%a.err

# runtime in HH:MM:SS format (DAYS-HH:MM:SS format)
#SBATCH --time=7-00:00:00

# job arrays
#SBATCH --array=0-1


# select partition
#SBATCH --partition=main


# load Julia module
module load julia/1.10.4

# set number of CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


# simulation parameter
N=100
DELTAS=(1.0 1.5)

DELTA_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

DELTA=${DELTAS[DELTA_INDEX]}

BONDDIM=70
KRAUSDIM=70

dt=0.5
nt=12000

# paths and file names
timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
ENV_PATH="/scratch/nguyed99/qcp-1d-julia/"
LOG_PATH="/scratch/nguyed99/qcp-1d-julia/logging"

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID
echo "slurm task ID = $SLURM_ARRAY_TASK_ID"
echo $N $DELTA $BONDDIM $KRAUSDIM $dt $nt
cat "${ENV_PATH}tests/test_open_dyn.jl"
cat "${ENV_PATH}src/lptn.jl" "${ENV_PATH}src/tebd.jl" "${ENV_PATH}src/models.jl" > "$LOG_PATH/${SLURM_ARRAY_JOB_ID}.func"

# launch Julia script
export JULIA_PROJECT=$ENV_PATH
julia ../tests/test_open_dyn.jl --N $N --DELTA $DELTA --BONDDIM $BONDDIM --KRAUSDIM $KRAUSDIM --dt $dt --nt $nt --JOBID $SLURM_ARRAY_JOB_ID 2>&1 > "$LOG_PATH/${SLURM_ARRAY_TASK_ID}.log"