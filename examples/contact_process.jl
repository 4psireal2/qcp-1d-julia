"""
L up to 50
"""

include("contact_process_model.jl")
include("../src/dmrgVMPO.jl")
include("../src/utility.jl")

using TensorKit
using Profile


### Model parameters
omega = 2.0;
gamma = 1.0;
d = 2; # physical dimension

### System parameters
N = 3;
bondDim = 1;

### basis states
basis0 = [1, 0];
basis1 = [0, 1];

### common operators
numberOps = constructNumberOps(N);


# ### common operators
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp,  ℂ^2 ,ℂ^2);

### initialize basis MPS
Id = [1 1; 1 1];
basis = fill(basis0*basis0', N);
initialMPS = initializeBasisMPS(N, basis, d=d);
initialMPS = orthonormalizeMPS(initialMPS);
# @show computeSiteExpVal_mps(initialMPS, numberOp)
@show computeSiteExpVal_vMPO(initialMPS, numberOp)

### initialize random MPS
initialMPS = initializeRandomMPS(N, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
lindblad1 = constructLindbladMPO(omega, gamma, N);
lindblad2 = constructLindbladDagMPO(omega, gamma, N);
lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);

# println("Run DMRG for bond dimension = $bondDim")
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = bondDim, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")

## run DMRG1
println("Run DMRG1 for bond dimension = $bondDim")
elapsed_time = @elapsed begin
gsMPS, gsEnergy = DMRG1(initialMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

# gsMPS_dag = Vector{TensorMap}(undef, N);
# for i = 1 : N
#     gsMPS_dag[i] = TensorMap(conj(permutedims(convert(Array, initialMPS[i]), (1,3,2,4))), codomain(initialMPS[i]), domain(initialMPS[i])); 
# end

# hermit_gsMPS = addMPSMPS(initialMPS, gsMPS_dag)
# hermit_gsMPS = orthonormalizeMPS(hermit_gsMPS);
# @show computeSiteExpVal_vMPO(hermit_gsMPS, numberOp)


nothing