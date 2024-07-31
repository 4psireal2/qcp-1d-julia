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
Id = [+1 0; 0 +1];
numberOpR = kron(numberOp, Id);
numberOpL = kron(Id, numberOp);
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

function createXMixed1()
    X = Vector{TensorMap}(undef, 1)
    X[1] = TensorMap(
        (1 / sqrt(2)) * [1 0; 0 1],
        ComplexSpace(1) ⊗ ComplexSpace(2),
        ComplexSpace(2) ⊗ ComplexSpace(1),
    )

    return X
end

basis0 = [1, 0];
basis1 = [0, 1];

# check purity of LPTN format
X = orthonormalizeX!(createXOnes(N; krausDim=3, bondDim=5));
normX = computeNorm(X);
@assert isapprox(normX, 1.0)
purityX = computePurity(X);
@assert isapprox(purityX, 1.0)

XSup = createXBasis(N, fill(1 / (sqrt(2)) * (basis0 + basis1), N));  # superposition state
@show computeNorm(XSup) # computeNorm(XSup) = 0.9999999999999992
@assert isapprox(computeNorm(XSup), 1.0)
@show computePurity(XSup) # computePurity(XSup) = 0.9999999999999984
@assert isapprox(computePurity(XSup), 1.0)
entSpec = computeEntSpec!(XSup);
entSpec[entSpec .== 0] .= 1e-5;
@show computevNEntropy(entSpec) # computevNEntropy(entSpec) = 4.605170852121907e-9

## A pure bipartite state is maximally entangled, if the reduced density matrix on either system is maximally mixed.
XMixed = createXMixed1();
@show computeNorm(XMixed) # computeNorm(XMixed) = 0.9999999999999998
@show computePurity(XMixed) # computePurity(XMixed) = 0.49999999999999983

# check norm of LPTN format
X = orthonormalizeX!(createXRand(N; krausDim=3, bondDim=5));
@show computeNorm(X) # computeNorm(X) = 0.9999999999999993
@assert isapprox(computeNorm(X), 1.0)
entSpec = computeEntSpec!(X);
entSpec[entSpec .== 0] .= 1e-5;
@show computevNEntropy(entSpec)
@assert computevNEntropy(entSpec) > 0.0

# check computeSiteExpVal
X = orthonormalizeX!(X; orthoCenter=1);
n_sites, n_t = computeSiteExpVal!(X, numberOp);
n_sites_test, n_t_test = computeSiteExpVal_test(X, numberOp);
@show n_sites, n_t # (n_sites, n_t) = ([0.8252846747812805, 0.5223583450165035, 0.3780963432477222, 0.30331262756876676, 0.37488695865563026], 0.48078778985398074)
@show n_sites_test, n_t_test # (n_sites_test, n_t_test) = ([0.8252846747812811, 0.5223583450165032, 0.3780963432477222, 0.3033126275687668, 0.3748869586556303], 0.48078778985398085)
@assert isapprox(n_sites, n_sites_test)
@assert isapprox(n_t, n_t_test)

XBasis1 = createXBasis(N, fill(basis1, N));
_, siteParticleNum = computeSiteExpVal!(XBasis1, numberOp);
@assert siteParticleNum == 1.0

# compute density-density correlation XXX: How to test this function?
correlation = zeros(N - 1);
for i in 2:N
    correlation[i - 1] = densDensCorr(i, XBasis1, numberOp)
end
@show correlation # correlation = [0.0, 0.0, 0.0, 0.0]

# check Kraus operator
diss =
    GAMMA * kron(annihilationOp, annihilationOp) - (1 / 2) * numberOpR - (1 / 2) * numberOpL;
diss = TensorMap(
    diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
);
diss = exp(dt * diss);
B = expDiss(GAMMA, dt);
@tensor checkKrausOp[-1 -2; -3 -4] := B'[-1, -3, 1] * conj(B'[-2, -4, 1]);
@show norm(checkKrausOp - diss) # norm(checkKrausOp - diss) = 1.5700924586837752e-16

@tensor checkKrausOp_1[-1 -2; -3 -4] := B'[-1, -3, 1] * B[-4, 1, -2];
@show norm(checkKrausOp_1 - diss) # norm(checkKrausOp_1 - diss) = 1.5700924586837752e-16

# check w/ QR/LQ and w\ QR/LQ for 1 time step
BONDDIM = 20;
KRAUSDIM = 10;
hamDyn = expHam(OMEGA, dt / 2);
dissDyn = expDiss(GAMMA, dt);
X_test = deepcopy(X);
X_t_test, ϵHTrunc_test, ϵDTrunc_test = TEBD_test(
    X_test, hamDyn, dissDyn, BONDDIM, KRAUSDIM; truncErr=truncErr
);
X_t, ϵHTrunc, ϵDTrunc = TEBD(X, hamDyn, dissDyn, BONDDIM, KRAUSDIM; truncErr=truncErr);

n_sites, n_t = computeSiteExpVal!(X_t, numberOp);
n_sites_test, n_t_test = computeSiteExpVal_test(X_t_test, numberOp);
@show n_sites, n_t # (n_sites, n_t) = ([0.7776988070902706, 0.49593016576010507, 0.36749675049502073, 0.29609656370833976, 0.3589129770462368], 0.4592270528199946)
@show n_sites_test, n_t_test # (n_sites_test, n_t_test) = ([0.7776988070902702, 0.4959301657601043, 0.3674967504950212, 0.2960965637083398, 0.35891297704623654], 0.45922705281999443)
@assert isapprox(n_sites, n_sites_test)
@assert isapprox(n_t, n_t_test)
@show maximum(ϵHTrunc_test), maximum(ϵDTrunc_test) # (maximum(ϵHTrunc_test), maximum(ϵDTrunc_test)) = (0.0011736754741027294, 9.237969111212845e-9)
@show maximum(ϵHTrunc), maximum(ϵDTrunc) # (maximum(ϵHTrunc), maximum(ϵDTrunc)) = (0.0011736754741027316, 9.237969092182622e-9)

nothing
