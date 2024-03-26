import LinearAlgebra: kron, norm, dot, tr

import Printf: @printf, @sprintf
import KrylovKit: eigsolve, Lanczos
import MPSKit: PeriodicArray
import TensorKit: @tensor, ⊗, fuse, isometry, leftorth, permute, randn, rightorth, space, 
                  truncerr, truncdim, tsvd, ComplexSpace, SVD, TensorMap, LQpos, RQpos

include("../utils/dmrg_vMPO.jl")
include("diss_ising_chain_model.jl")


# Model parameters
GAMMA = 1.0; V = 5.0; OMEGA = 1.5; DELTA = 1.0;

# System parameters
N=3;


println("Check for Hermiticity of Lindbladian for N=3")
N = 3;
lindblad1 = constructLindbladMPO(GAMMA, V, OMEGA, DELTA, N);
lindblad2 = constructLindbladDagMPO(GAMMA, V, OMEGA, DELTA, N);
lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);
boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
@tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := boundaryL[1] * lindbladHermitian[1][1, -1, -2, -7, -8, 2] *
                                                                               lindbladHermitian[2][2, -3, -4, -9, -10, 3] *
                                                                               lindbladHermitian[3][3, -5, -6, -11, -12, 4] *
                                                                               boundaryR[4];
lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));
@show norm(lindbladMatrix - lindbladMatrix') == 0.0

### basis states
basis0 = [1, 0];
basis1 = [0, 1];
basis = fill(kron(basis0, basis0), N);
initialMPS = initializeBasisMPS(N, basis, d=d);
initialMPS = orthonormalizeMPS(initialMPS);


println("Compute magnetisation")
Mzs = Vector{TensorMap}(undef, N);
Mz = 0.5 * (kron([+1 0 ; 0 -1], [1 0; 0 1]) + kron([1 0; 0 1], [+1 0 ; 0 -1]));
for i = 1 : N
    Mzs[i] = TensorMap(Mz, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
end

@show mZSite = computeSiteExpVal(initialMPS, Mzs)

println("Compute staggered magnetisation")
MzStags = multiplyMPOMPO(constructMzStagsDag(N), constructMzStags(N));
MzStagsOp = Vector{TensorMap}(undef, N);
for i = 1 : N
    @tensor MzStagsOp[i][-1 -2; -3 -4] := boundaryL[1] * MzStags[i][1, -1, -2, -3, -4, 2] * boundaryR[2];
end

@show mZStagSite = sum(computeSiteExpVal(initialMPS, MzStagsOp))/N;
nothing