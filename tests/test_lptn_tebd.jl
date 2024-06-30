include("../src/lptn.jl")
include("../src/tebd.jl")

using LinearAlgebra
using TensorKit


N = 5;
OMEGA = 6.0;
GAMMA = 1.0;


TAU = 0.5;
BONDDIM = 5;
KRAUSDIM = 3;
truncErr = 1e-6;
nTimeSteps = 3;

## common operators
annihilationOp = [0 1; 0 0];
numberOp = [0 0; 0 1];
Id = [+1 0 ; 0 +1];
numberOpR = kron(numberOp, Id);
numberOpL = kron(Id, numberOp);

# check norm of LPTN format
X = orthonormalizeX(createXOnes(N, krausDim=3, bondDim=5));
@show computeNorm(X)
@assert isapprox(computeNorm(X), 1.0)
@show computePurity(X)
@assert isapprox(computePurity(X), 1.0)

# check purity of LPTN format
function createXMixed1()

    X = Vector{TensorMap}(undef, 1);
    X[1] = TensorMap( (1/ sqrt(2)) * [1 0; 0 1], ComplexSpace(1) ⊗ ComplexSpace(2),  ComplexSpace(2) ⊗ ComplexSpace(1));

    return X
end


# function createXMixed2()

#     X = Vector{TensorMap}(undef, 2);
#     X[1] = TensorMap( (1/ sqrt(2)) * [1 0; 0 1], ComplexSpace(1) ⊗ ComplexSpace(2),  ComplexSpace(2) ⊗ ComplexSpace(1));
#     X[2] = TensorMap( (1/ sqrt(2)) * [1 0; 0 1], ComplexSpace(1) ⊗ ComplexSpace(2),  ComplexSpace(d) ⊗ ComplexSpace(1));

#     return X
# end



basis0 = [1, 0];
basis1 = [0, 1];

XSup = createXBasis(N, fill(1/(sqrt(2)) * (basis0 + basis1), N));  # superposition state
@show computeNorm(XSup)
@assert isapprox(computeNorm(XSup), 1.0)
@show computePurity(XSup)
@assert isapprox(computePurity(XSup), 1.0)
@show computeEntSpec(XSup)


# A pure bipartite state is maximally entangled, if the reduced density matrix on either system is maximally mixed.
XMixed  = createXMixed1();
@show computeNorm(XMixed)
@show computePurity(XMixed)

# compute number density
XBasis1 = createXBasis(N, fill(basis1, N));
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);
_, siteParticleNum = computeSiteExpVal(XBasis1, numberOp);
@assert siteParticleNum == 1.0

# compute density-density correlation XXX: How to test this function?
correlation = zeros(N-1);
for i in 2:N
    correlation[i-1] = densDensCorr(i, XBasis1, numberOp)
end
@show correlation

# check Kraus operator
diss = GAMMA * kron(annihilationOp, annihilationOp) - (1/2) * numberOpR - (1/2) * numberOpL;
diss = TensorMap(diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)');
diss = exp(TAU * diss);
B = expDiss(GAMMA, TAU);
@tensor checkKrausOp[-1 -2; -3 -4] := B'[-1, -3, 1] * conj(B'[-2, -4, 1]);
@show norm(checkKrausOp - diss)

@tensor checkKrausOp_1[-1 -2; -3 -4] := B'[-1, -3, 1] * B[-4, 1, -2];
@show norm(checkKrausOp_1 - diss)

# run dynamics simulation
hamDyn = expHam(OMEGA, TAU/2);
dissDyn = expDiss(GAMMA, TAU);
X_t = X;
for i = 1 : nTimeSteps
    global  X_t
    X_t = TEBD(X_t, hamDyn, dissDyn, BONDDIM, KRAUSDIM, truncErr);
end

nothing