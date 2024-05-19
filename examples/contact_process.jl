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
N = 10;
bondDim = 8;

### basis states
basis0 = [1, 0];
basis1 = [0, 1];

### common operators
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp,  ℂ^2 ,ℂ^2);

### initialize basis MPS
Id = [1 1; 1 1];
basis = fill(basis0*basis0' + basis1*basis1', N);
initialMPS = initializeRandomMPS(N, 2, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);





# hermit_initialMPS = addMPSMPS(computeRhoDag(initialMPS), initialMPS);
# hermit_initialMPS = orthonormalizeMPS(hermit_initialMPS);


### initialize random MPS
# initialMPS = initializeRandomMPS(N, bonddim=bondDim);
# initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
lindblad = constructLindbladMPO(omega, gamma, N);
lindblad_dag = constructLindbladDagMPO(omega, gamma, N);
lindbladHermitian = multiplyMPOMPO(lindblad, lindblad_dag);

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



hermit_gsMPS = addMPSMPS(gsMPS, computeRhoDag(gsMPS))
# hermit_gsMPS = orthonormalizeMPS(hermit_gsMPS);
@show computeSiteExpVal_vMPO(hermit_gsMPS, numberOp)

@show computeSiteExpVal_mps(hermit_gsMPS, numberOp)


# expectation values
# basis = fill(basis0*basis0', N);
# darkState = initializeBasisMPS(N, basis, d=d);
# darkState = orthonormalizeMPS(darkState);
# @show computeSiteExpVal_vMPO(darkState, numberOp)

# @show computeSiteExpVal_mps(darkState, numberOp)

# basis = fill(basis0*basis0' + basis1*basis1', N);
# mixed_state = initializeBasisMPS(N, basis);
# # mixed_state = orthonormalizeMPS(mixed_state);
# @show computeSiteExpVal_vMPO(mixed_state, numberOp)

# @show computeSiteExpVal_mps(mixed_state, numberOp)
nothing