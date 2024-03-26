"""
L up to 50
"""

using TensorKit
using Profile

include("../utils/dmrg_vMPO.jl")
include("contact_process_model.jl")


### Model parameters
omega = 5.0;
gamma = 1.0;
d = 2; # physical dimension

### System parameters
N = 10;
bondDim = 1;

### basis states
basis0 = [1, 0];
basis1 = [0, 1];

### initialize basis MPS
basis = fill(kron(basis1, basis1), N);
# initialMPS = initializeBasisMPS(N, basis, d=d);
# initialMPS = orthonormalizeMPS(initialMPS);


### initialize random MPS
initialMPS = initializeRandomMPS(N);
initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
# lindblad1 = constructLindbladMPO(omega, gamma, N);
# lindblad2 = constructLindbladDagMPO(omega, gamma, N);
# lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);

### run DMRG
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = 1, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
# # @printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

### run DMRG for larger bondDim
# println("Run DMRG for larger bondDim")
# # maxIterations = variable for Eval Solver
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(gsMPS, lindbladHermitian, bondDim = 4, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
# # @printf("Ground state energy per site E = %0.6f", gsEnergy / N)

### compute particle numbers
numberOps = constructNumberOps(N);
# particleNums = computeSiteExpVal(gsMPS, numberOps);