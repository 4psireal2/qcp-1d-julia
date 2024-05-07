include("ising_chain_model.jl")
include("../src/utility.jl")
include("../src/dmrgVMPO.jl")

using LinearAlgebra
using Plots
using Printf
using Statistics

using TensorKit
# import LinearAlgebra: eigenvals

### Model parameters
GAMMA = 1.0; V = 5.0; OMEGA = 1.5; DELTA = -2.4;

### System parameters
N=10;
D=50;
bondDim = 16;

println("Check for Hermiticity of Lindbladian for N=3")
N = 3;

# DELTA = range(-4, 6, length=25);

# lindblad = constructLindbladMPO(GAMMA, V, OMEGA, DELTA, N);
# lindblad_dag = constructLindbladDagMPO(GAMMA, V, OMEGA, DELTA, N);
# lindbladHermitian = multiplyMPOMPO(lindblad_dag, lindblad);
# boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
# boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
# @tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := boundaryL[1] * lindbladHermitian[1][1, -1, -2, -7, -8, 2] *
#                                                                                lindbladHermitian[2][2, -3, -4, -9, -10, 3] *
#                                                                                lindbladHermitian[3][3, -5, -6, -11, -12, 4] *
#                                                                                boundaryR[4];
# lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));
# @show norm(lindbladMatrix - lindbladMatrix') == 0.0

# println("Check for exact diagonalization of Lindbladian for N=3")
# Evals, Evectors = eigen(lindbladMatrix)
# @show Evals

### Plot spectral gap
# Evals, Evectors = eigen(lindbladMatrix)
# sort!(Evals);
# @show Evals[2] - Evals[1]


### basis states
# basis0 = [1, 0];
# basis1 = [0, 1];
# basis = fill(kron(basis0, basis0), N);
# initialMPS = initializeBasisMPS(N, basis);
# initialMPS = orthonormalizeMPS(initialMPS);


# ### DMRG
N = 10;
lindblad = constructLindbladMPO(GAMMA, V, OMEGA, DELTA, N);
lindblad_dag = constructLindbladDagMPO(GAMMA, V, OMEGA, DELTA, N);
lindbladHermitian = multiplyMPOMPO(lindblad_dag, lindblad);

initialMPS = initializeRandomMPS(N, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

# println("Run DMRG for bond dimension = $bondDim")
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = bondDim, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

println("Run DMRG1 for bond dimension = $bondDim")
elapsed_time = @elapsed begin
gsMPS, gsEnergy = DMRG1(initialMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

# println("Compute magnetisation")
# Mzs = Vector{TensorMap}(undef, N);
# Mz = 0.5 * (kron([+1 0 ; 0 -1], [1 0; 0 1]) + kron([1 0; 0 1], [+1 0 ; 0 -1]));
# for i = 1 : N
#     Mzs[i] = TensorMap(Mz, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
# end

# Mxs = Vector{TensorMap}(undef, N);
# Mx = 0.5 * (kron([0 +1 ; +1 0], [1 0; 0 1]) + kron([1 0; 0 1], [0 +1 ; +1 0]));
# for i = 1 : N
#     Mxs[i] = TensorMap(Mx, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
# end

# Mys = Vector{TensorMap}(undef, N);
# My = 0.5 * (kron([0 -1im ; +1im 0], [1 0; 0 1]) + kron([1 0; 0 1], [0 -1im ; +1im 0]));
# for i = 1 : N
#     Mys[i] = TensorMap(My, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
# end

# @show mZSite = computeSiteExpVal(initialMPS, Mzs)

# println("Compute staggered magnetisation")
# MzStags = multiplyMPOMPO(constructMzStagsDag(N), constructMzStags(N));
# MzStagsOp = Vector{TensorMap}(undef, N);
# for i = 1 : N
#     @tensor MzStagsOp[i][-1 -2; -3 -4] := boundaryL[1] * MzStags[i][1, -1, -2, -3, -4, 2] * boundaryR[2];
# end

# @show mZStagSite = mean(computeSiteExpVal(initialMPS, MzStagsOp));

# println("Compute purity of mixed state")
# @show purity = computePurity(initialMPS);


# println("Run DMRG for bond dimension = 1")
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = 16, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
# @printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

# ### Check for unphysical solutions
# if gsEnergy < 1e-12

#     if  -N <= sum(computeSiteExpVal(initialMPS, Mxs)) <= N || 
#         -N <= sum(computeSiteExpVal(initialMPS, Mys)) <= N || 
#         -N <= sum(computeSiteExpVal(initialMPS, Mzs)) <= N
    
#         println("Run DMRG for bond dimension = 2")
#         elapsed_time = @elapsed begin
#             gsMPS, gsEnergy = DMRG2(gsMPS, lindbladHermitian, bondDim = 2, truncErr = 1e-6, convTolE = 1e-5, maxIterations=1, verbosePrint = true);
#         end
#         println("Elapsed time for DMRG2: $elapsed_time seconds")
#         @printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)
#     else
#         throw(ErrorException("Solution is unphysical!"))
#     end

# else
#     throw(ErrorException("The exact solution has zero eigenvalue!"))
# end
nothing