include("../src/lptn.jl")
include("../src/tebd.jl")

using LinearAlgebra
using TensorKit


N = 5;
OMEGA = 6.0;
GAMMA = 1.0;


dt = 0.05;
BONDDIM = 5;
KRAUSDIM = 3;
truncErr = 1e-6;
nTimeSteps = 1;

## common operators
numberOp = [0 0; 0 1];
annihilationOp = [0 1; 0 0];
Id = [+1 0 ; 0 +1];
numberOpR = kron(numberOp, Id);
numberOpL = kron(Id, numberOp);
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

function createXMixed1()

    X = Vector{TensorMap}(undef, 1);
    X[1] = TensorMap( (1/ sqrt(2)) * [1 0; 0 1], ComplexSpace(1) ⊗ ComplexSpace(2),  ComplexSpace(2) ⊗ ComplexSpace(1));

    return X
end

basis0 = [1, 0];
basis1 = [0, 1];


# check purity of LPTN format
X = orthonormalizeX!(createXOnes(N, krausDim=3, bondDim=5));
normX = computeNorm(X);
@assert isapprox(normX, 1.0)
purityX = computePurity(X);
@assert isapprox(purityX, 1.0)

XSup = createXBasis(N, fill(1/(sqrt(2)) * (basis0 + basis1), N));  # superposition state
@show computeNorm(XSup)
@assert isapprox(computeNorm(XSup), 1.0)
@show computePurity(XSup)
@assert isapprox(computePurity(XSup), 1.0)
@show computeEntSpec!(XSup)

## A pure bipartite state is maximally entangled, if the reduced density matrix on either system is maximally mixed.
XMixed  = createXMixed1();
@show computeNorm(XMixed)
@show computePurity(XMixed)

# check norm of LPTN format
X = orthonormalizeX!(createXRand(N, krausDim=3, bondDim=5));
@show computeNorm(X)
@assert isapprox(computeNorm(X), 1.0)

# check computeSiteExpVal
n_sites, n_t = computeSiteExpVal!(X, numberOp);
n_sites_test, n_t_test = computeSiteExpVal_test(X, numberOp);
@show n_sites, n_t
@show n_sites_test, n_t_test
@assert isapprox(n_sites, n_sites_test)
@assert isapprox(n_t, n_t_test)

XBasis1 = createXBasis(N, fill(basis1, N));
_, siteParticleNum = computeSiteExpVal!(XBasis1, numberOp);
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
diss = exp(dt * diss);
B = expDiss(GAMMA, dt);
@tensor checkKrausOp[-1 -2; -3 -4] := B'[-1, -3, 1] * conj(B'[-2, -4, 1]);
@show norm(checkKrausOp - diss)

@tensor checkKrausOp_1[-1 -2; -3 -4] := B'[-1, -3, 1] * B[-4, 1, -2];
@show norm(checkKrausOp_1 - diss)

# check w/ QR/LQ and w\ QR/LQ for 1 time step
BONDDIM=20;
KRAUSDIM=10;
hamDyn = expHam(OMEGA, dt/2);
dissDyn = expDiss(GAMMA, dt);
X_test = deepcopy(X);
X_t_test, ϵHTrunc_test, ϵDTrunc_test = TEBD_test(X_test, hamDyn, dissDyn, BONDDIM, KRAUSDIM, truncErr=truncErr);
X_t, ϵHTrunc, ϵDTrunc = TEBD(X, hamDyn, dissDyn, BONDDIM, KRAUSDIM, truncErr=truncErr);

n_sites, n_t = computeSiteExpVal!(X_t, numberOp);
n_sites_test, n_t_test = computeSiteExpVal_test(X_t_test, numberOp);
@show n_sites, n_t
@show n_sites_test, n_t_test
@assert isapprox(n_sites, n_sites_test)
@assert isapprox(n_t, n_t_test)
@show maximum(ϵHTrunc_test), maximum(ϵDTrunc_test)
@show maximum(ϵHTrunc), maximum(ϵDTrunc)

nothing