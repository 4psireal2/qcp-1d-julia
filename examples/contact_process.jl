"""
L up to 50
"""

include("contact_process_model.jl")
include("../src/dmrgVMPO.jl")
include("../src/utility.jl")

using Statistics
using TensorKit
using Profile


### Model parameters
omega = 6.5;
gamma = 1.0;
d = 2; # physical dimension

### System parameters
N = 10;
bondDim = 10;

### common operators
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp,  ℂ^2 ,ℂ^2);

### initialize random MPS
initialMPS = initializeRandomMPS(N, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
lindblad = constructLindbladMPO(omega, gamma, N);
lindblad_dag = constructLindbladDagMPO(omega, gamma, N);
lindbladHermitian = multiplyMPOMPO(lindblad, lindblad_dag);

### DMRG1
println("Run DMRG1 for bond dimension = $bondDim")
elapsed_time = @elapsed begin
gsMPS, gsEnergy = DMRG1(initialMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)


hermit_gsMPS = addMPSMPS(gsMPS, computeRhoDag(gsMPS));

number_density_vMPO = computeSiteExpVal_vMPO(hermit_gsMPS, numberOp);
number_density_mps = computeSiteExpVal_mps(hermit_gsMPS, numberOp);
@show number_density_vMPO
@show number_density_mps
@show mean(number_density_vMPO)
@show mean(number_density_mps)

nothing