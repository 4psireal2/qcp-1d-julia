include("../examples/contact_process_model.jl")
include("../src/dmrg.jl")
include("../src/vmPO.jl")
using Printf
using LinearAlgebra
using Random

N = 5;
d = 2;
OMEGA = 6.0;
GAMMA = 1.0;

# test average site expectation value
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

basis0 = [1, 0];
basis1 = [0, 1];

basis = fill(basis0 * basis0', N);
basisMPS = initializeBasisMPS(N, basis; d=d);
basisMPS = orthonormalizeMPS(basisMPS);
@assert computeSiteExpVal_vMPO(basisMPS, numberOp) == zeros(N)

basis = fill(basis0 * basis1', N);
basisMPS = initializeBasisMPS(N, basis; d=d);
basisMPS = orthonormalizeMPS(basisMPS);
@assert computeSiteExpVal_vMPO(basisMPS, numberOp) == zeros(N)

basis = fill(basis1 * basis0', N);
basisMPS = initializeBasisMPS(N, basis; d=d);
basisMPS = orthonormalizeMPS(basisMPS);
@assert computeSiteExpVal_vMPO(basisMPS, numberOp) == zeros(N)

basis = fill(basis1 * basis1', N);
basisMPS = initializeBasisMPS(N, basis; d=d);
basisMPS = orthonormalizeMPS(basisMPS);
@assert computeSiteExpVal_vMPO(basisMPS, numberOp) == ones(N)

basis = fill(basis0 * basis0' + basis1 * basis1', N);
basisMPS = initializeBasisMPS(N, basis; d=d);
@assert isapprox(computeSiteExpVal_vMPO(basisMPS, numberOp), 1 / 2 * ones(N))

# test Lindbladian
## test Hermiticity
N = 3;
lindblad = constructLindbladMPO(OMEGA, GAMMA, N);
lindblad_dag = constructLindbladDagMPO(OMEGA, GAMMA, N);

boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
lindbladHermitian = multiplyMPOMPO(lindblad, lindblad_dag);
@tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] :=
    boundaryL[1] *
    lindbladHermitian[1][1, -1, -2, -7, -8, 2] *
    lindbladHermitian[2][2, -3, -4, -9, -10, 3] *
    lindbladHermitian[3][3, -5, -6, -11, -12, 4] *
    boundaryR[4];

# @tensor lindbladCheck[-1 -2 -3 -4; -5 -6 -7 -8] := boundaryL[1] * lindblad[1][1, -1, -2, -5, -6, 2] * lindblad[2][2, -3, -4, -7, -8, 3] * boundaryR[3];

lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));
lindbladMatrix /= norm(lindbladMatrix);

@assert norm(lindbladMatrix - lindbladMatrix') == 0.0

## test exact diagonalization
Evals, Evectors = eigen(lindbladMatrix);
@assert sort(Evals)[1] == 0.0

## test steady state =? dark state
basis = fill(basis0 * basis0', N);
darkState = initializeBasisMPS(N, basis; d=d);
darkState = orthonormalizeMPS(darkState);
@tensor darkState[-1 -2 -3 -4 -5 -6] :=
    boundaryL[1] *
    darkState[1][1, -1, -2, 2] *
    darkState[2][2, -3, -4, 3] *
    darkState[3][3, -5, -6, 4] *
    boundaryR[4];
darkState_vec = reshape(convert(Array, darkState), (64,));
@assert darkState_vec == Evectors[1, :]

## test transpose complex
rng = MersenneTwister(1234);

randomMPS = Vector{TensorMap}(undef, N);
randImMat = randn(rng, ComplexF32, (2, 2))
randomMPS[1] = TensorMap(
    randImMat, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)', ComplexSpace(1)
);
for i in 2:(N - 1)
    randImMat = randn(rng, ComplexF32, (2, 2))
    randomMPS[i] = TensorMap(
        randImMat, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)', ComplexSpace(1)
    )
end
randImMat = randn(rng, ComplexF32, (2, 2))
randomMPS[N] = TensorMap(
    randn, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)', ComplexSpace(1)
);

boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))

@tensor randomMPSmat[-1 -2 -3; -4 -5 -6] :=
    boundaryL[1] *
    randomMPS[1][1, -1, -4, 2] *
    randomMPS[2][2, -2, -5, 3] *
    randomMPS[3][3, -3, -6, 4] *
    boundaryR[4];
randomMPSmat = reshape(convert(Array, randomMPSmat), (8, 8));

randomMPS_dag = Vector{TensorMap}(undef, N);
for i in 1:N
    randomMPS_dag[i] = TensorMap(
        conj(permutedims(convert(Array, randomMPS[i]), (1, 3, 2, 4))),
        codomain(randomMPS[i]),
        domain(randomMPS[i]),
    )
end

boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))

@tensor randomMPSdag_mat[-1 -2 -3; -4 -5 -6] :=
    boundaryL[1] *
    randomMPS_dag[1][1, -1, -4, 2] *
    randomMPS_dag[2][2, -2, -5, 3] *
    randomMPS_dag[3][3, -3, -6, 4] *
    boundaryR[4];
randomMPSdag_mat = reshape(convert(Array, randomMPSdag_mat), (8, 8));

@assert randomMPSdag_mat == randomMPSmat'

## test adding mps
hermit_MPS = addMPSMPS(randomMPS, randomMPS_dag);
boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
@tensor hermit_mat[-1 -2 -3; -4 -5 -6] :=
    boundaryL[1] *
    hermit_MPS[1][1, -1, -4, 2] *
    hermit_MPS[2][2, -2, -5, 3] *
    hermit_MPS[3][3, -3, -6, 4] *
    boundaryR[4];
hermit_mat = reshape(convert(Array, hermit_mat), (8, 8));
@assert norm(hermit_mat - hermit_mat') == 0.0

basis = fill(basis1 * basis1', N);
basisMPS = initializeBasisMPS(N, basis; d=d);

non_hermitian = addMPSMPS(computeRhoDag(basisMPS), -1 * basisMPS);
non_hermitian = orthogonalizeMPS(non_hermitian);
norm_non_hermitian = real(tr(non_hermitian[1]' * non_hermitian[1]));
@assert norm_non_hermitian == 0.0

hermitian = addMPSMPS(computeRhoDag(basisMPS), basisMPS);
hermitian = orthonormalizeMPS(hermitian);
norm_hermitian = real(tr(hermitian[1]' * hermitian[1]));
@assert norm_hermitian == 1.0

nothing
