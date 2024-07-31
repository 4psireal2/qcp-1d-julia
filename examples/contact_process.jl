"""
L up to 50
"""

include("contact_process_model.jl")
include("../src/dmrg.jl")
include("../src/dmrg_excited.jl")
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
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

### initialize random MPS
initialMPS = initializeRandomMPS(N; bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
lindblad = constructLindbladMPO(omega, gamma, N);
lindblad_dag = constructLindbladDagMPO(omega, gamma, N);
lindbladHermitian = multiplyMPOMPO(lindblad, lindblad_dag);

### DMRG1
println("INFO: Run DMRG1 for bond dimension = $bondDim, N = $N, Ω = $omega")
elapsed_time = @elapsed begin
    gsMPS, gsEnergy = DMRG1(
        initialMPS, lindbladHermitian; convTolE=1e-6, maxIterations=1, verbosePrint=true
    )
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

println("INFO: Compute contribution of non-Hermitian part")
non_hermitian = addMPSMPS(computeRhoDag(gsMPS), -1 * gsMPS);

non_hermitian = orthogonalizeMPS(non_hermitian);
norm_non_hermitian = real(tr(non_hermitian[1]' * non_hermitian[1]));
@show norm_non_hermitian

hermit_gsMPS = addMPSMPS(computeRhoDag(gsMPS), gsMPS);

number_density_vMPO = computeSiteExpVal_vMPO(hermit_gsMPS, numberOp);
number_density_mps = computeSiteExpVal_mps(hermit_gsMPS, numberOp);
@show number_density_vMPO
@show number_density_mps
@show mean(number_density_vMPO)
@show mean(number_density_mps)

# ### DMRG1 excited
# println("INFO: Search for first excited state")
# first_exc, energy_first_exc = find_excitedstate(initialMPS, lindbladHermitian, [hermit_gsMPS], DMRG1_params())

# println("INFO: Compute contribution of non-Hermitian part")
# non_hermitian = addMPSMPS(computeRhoDag(first_exc), -1 * first_exc);
# non_hermitian = orthogonalizeMPS(non_hermitian);
# norm_non_hermitian = real(tr(non_hermitian[1]' * non_hermitian[1]));
# @show norm_non_hermitian

# hermit_first_exc = addMPSMPS(computeRhoDag(first_exc), first_exc);

# number_density_vMPO = computeSiteExpVal_vMPO(hermit_first_exc, numberOp);
# number_density_mps = computeSiteExpVal_mps(hermit_first_exc, numberOp);
# @show number_density_vMPO
# @show number_density_mps
# @show mean(number_density_vMPO)
# @show mean(number_density_mps)

nothing
