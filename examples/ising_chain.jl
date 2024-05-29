include("ising_chain_model.jl")
include("../src/utility.jl")
include("../src/dmrg.jl")

using LinearAlgebra
using Plots
using Printf
using Statistics

using TensorKit

### Model parameters
GAMMA = 1.0; V = 5.0; OMEGA = 1.5; DELTA = 0.0;

### System parameters
N=10;
bondDim = 10;


### basis states
basis0 = [1, 0];
basis1 = [0, 1];
basis = fill(basis0*basis0', N);
# initialMPS = initializeBasisMPS(N, basis);
# initialMPS = orthonormalizeMPS(initialMPS);


### construct Lindbladian MPO 
lindblad = constructLindbladMPO(GAMMA, V, OMEGA, DELTA, N);
lindblad_dag = constructLindbladDagMPO(GAMMA, V, OMEGA, DELTA, N);
lindbladHermitian = multiplyMPOMPO(lindblad, lindblad_dag);

initialMPS = initializeRandomMPS(N, bonddim=bondDim);
initialMPS = orthonormalizeMPS(initialMPS);

### DMRG2
# println("Run DMRG2 for bond dimension = $bondDim, Ω = $omega")
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = 1*bondDim, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
# @printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

# factor_n = 8;
# while gsEnergy > 1e-7
#     println("Run DMRG2 for bond dimension = $factor_n")
#     elapsed_time = @elapsed begin
#         global gsMPS
#         global  gsEnergy
#         gsMPS, gsEnergy = DMRG2(gsMPS, lindbladHermitian, bondDim = factor_n*bondDim, truncErr = 1e-6, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
#         global factor_n
#         factor_n *=2
#     end
#     println("Elapsed time for DMRG2: $elapsed_time seconds")
#     @printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)
# end

### DMRG1
println("Run DMRG1 for bond dimension = $bondDim")
elapsed_time = @elapsed begin
gsMPS, gsEnergy = DMRG1(initialMPS, lindbladHermitian, convTolE = 1e-6, maxIterations=1, verbosePrint = true);
end
println("Elapsed time for DMRG1: $elapsed_time seconds")
@printf("Ground state energy per site E = %0.6f\n", gsEnergy / N)

hermit_gsMPS = 1/2 * addMPSMPS(gsMPS, computeRhoDag(gsMPS))

println("Compute magnetisation")
sigmaZ = 0.5 * [+1 0 ; 0 -1];
sigmaZ = TensorMap(sigmaZ,  ℂ^2 ,ℂ^2);
local_polarization_z_vMPO = computeSiteExpVal_vMPO(hermit_gsMPS, sigmaZ);
local_polarization_z_mps = computeSiteExpVal_mps(hermit_gsMPS, sigmaZ);
@show local_polarization_z_vMPO
plot(range(1,10, length=10), local_polarization_z_mps)


## local_polarization_z_mps = [-0.4910078429417828, -0.49580364217264344, -0.49581414869660134, -0.4958142146153981, 
## -0.495814348331521, -0.4958141751343449, -0.4958141844301206, -0.4958141723583373, -0.49580382311610816, -0.4910078120963935] # Δ=-4.0

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