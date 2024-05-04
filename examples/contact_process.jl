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
bondDim = 16;

### basis states
basis0 = [1, 0];
basis1 = [0, 1];

### common operators
numberOps = constructNumberOps(N);


# ### common operators
numberOp = (kron([0 0; 0 1], [1 0; 0 1]) + kron([1 0; 0 1], [0 0; 0 1]));
numberOp = TensorMap(numberOp, ℂ^1 ⊗ ℂ^2 ⊗ conj(ℂ^2), ℂ^2 ⊗ conj(ℂ^2) ⊗ ℂ^1);

### initialize basis MPS
basis = fill(kron(basis1, basis0), N);
initialMPS = initializeBasisMPS(N, basis, d=d);
initialMPS = orthonormalizeMPS(initialMPS);
@show computeSiteExpVal_mps(initialMPS, numberOp)
@show computeSiteExpVal_vMPO(initialMPS, numberOp)

### initialize random MPS
initialMPS = initializeRandomMPS(N, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

### construct Lindbladian MPO for quantum contact process
lindblad1 = constructLindbladMPO(omega, gamma, N);
lindblad2 = constructLindbladDagMPO(omega, gamma, N);
lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);

### run DMRG1
println("Run DMRG1 for bond dimension = $bondDim")
elapsed_time = @elapsed begin
gsMPS, gsEnergy = DMRG1(initialMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

@show computeSiteExpVal_vMPO(gsMPS, numberOp)

# # compute Hermitian part of MPS
# hermitgsMPS = Vector{TensorMap}(undef, N);
# for i = 1 : N
#     mps_dag_i = TensorMap(conj(convert(Array, gsMPS[i])), codomain(gsMPS[i]), domain(gsMPS[i])); 
#     hermitgsMPS[i] = mps_dag_i + gsMPS[i]
# end
# hermitgsMPS /= 2


# @show computeExpVal(hermitgsMPS, numberOps)

# for bondDim in bondDims

#     ### Check for unphysical solutions
#     if 0 <= sum(particleNums) <= 2*N
#         println("Run DMRG for bond dimension = $bondDim")
#         elapsed_time = @elapsed begin
#         gsMPSNew, gsEnergyNew = DMRG1(gsMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true)
#         end
#         println("Elapsed time for DMRG1: $elapsed_time seconds")
#         @printf("Ground state energy per site E = %0.6f", gsEnergyNew / N)

#         push!(gsEnergies, gsEnergy)
#         @assert gsEnergies[end-1] <= gsEnergies[end] # assert energy convergence

#     else
#         throw(ErrorException("Solution is unphysical!"))
#     end
# end    


### eigenvalue check
# using LinearAlgebra

# N = 3;
# physicalDim = 2;

# # initialize MPS
# initialMPS = initializeRandomMPS(N);
# orthonormalizeMPS(initialMPS);

# # construct Lindbladian MPO for quantum contact process
# lindblad1 = constructLindbladMPO(6.0, 1.0, N);
# lindblad2 = constructLindbladDagMPO(6.0, 1.0, N);

# boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
# boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
# lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);
# @tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := boundaryL[1] * lindbladHermitian[1][1, -1, -2, -7, -8, 2] *
#                                                                                lindbladHermitian[2][2, -3, -4, -9, -10, 3] *
#                                                                                lindbladHermitian[3][3, -5, -6, -11, -12, 4] *
#                                                                                boundaryR[4];

# lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));
# @show eigvals(lindbladMatrix)[1];

nothing