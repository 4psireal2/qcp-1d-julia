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

# check purity of LPTN format
basis0 = [1, 0];
basis1 = [0, 1];

XSup = createXBasis(1, fill(1/(sqrt(2)) * (basis0 + basis1), 1)); 
@show computeNorm(XSup)
@show computePurity(XSup)

# A pure bipartite state is maximally entangled, if the reduced density matrix on either system is maximally mixed.
XMixed  = createXMixed1();
@show computeNorm(XMixed)
@show computePurity(XMixed)

# compute number density

XBasis1 = createXBasis(N, fill(basis1, N));
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);
siteParticleNum = computeSiteExpVal(XBasis1, numberOp);
@assert siteParticleNum == 1.0

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