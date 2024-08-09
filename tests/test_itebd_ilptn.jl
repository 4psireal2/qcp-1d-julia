using LinearAlgebra
using TensorKit

# include("../src/itebd.jl")
# include("../src/models.jl")
include("../src/ilptn.jl")


# Test: canonical form
# bond1 = odd bond - bondTensor = Le - Go - Lo - Ge - Le
# bond2 = even bond - bondTensor = Lo - Ge - Le - Go - Lo

d = 2;
bondDim = 2;
krausDim = 3
bondDimTrunc = 5;
maxiter = 1000;
tol = 1e-6;

Go = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim), ComplexSpace(d) ⊗ ComplexSpace(bondDim));
Ge = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim), ComplexSpace(d) ⊗ ComplexSpace(bondDim));
Lo = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));
Le = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));


initGuess = Matrix(I, bondDim, bondDim) ./ bondDim;
transferOpLBond1 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim));
transferOpRBond2 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim));

transferOpLBond1, transferOpLBond2 = leftContraction!(transferOpLBond1, Le, Go, Lo, Ge)
transferOpRBond2, transferOpRBond1 = rightContraction!(transferOpRBond2, Le, Go, Lo, Ge)

Ge, Le, Go, gaugeLEven, gaugeREven = orthonormalizeiLPTN(
    transferOpLBond1, transferOpRBond1, Lo, Ge, Le, Go
)
Go, Lo, Ge, gaugeLOdd, gaugeROdd = orthonormalizeiLPTN(
    transferOpLBond2, transferOpRBond2, Le, Go, Lo, Ge
)

@tensor transferOpLBond1[-1; -2] :=
    gaugeLEven'[-1, 1] * transferOpLBond1[1, 2] * gaugeLEven[2, -2]
@show transferOpLBond1

# @tensor transferOpLBond1[-2; -1] :=
#     gaugeLEven[1, -1] * transferOpLBond1[2, 1] * gaugeLEven'[2, -2]
# @show transferOpLBond1

@tensor transferOpRBond1[-1; -2] :=
    gaugeREven[-1, 1] * transferOpRBond1[1, 2] * gaugeREven'[2, -2]
@show transferOpRBond1

@tensor transferOpLBond2[-1; -2] :=
    gaugeLOdd'[-1, 1] * transferOpLBond2[1, 2] * gaugeLOdd[2, -2]
@show transferOpLBond2

@tensor transferOpRBond2[-1; -2] :=
    gaugeROdd[-1, 1] * transferOpRBond2[1, 2] * gaugeROdd'[2, -2]
@show transferOpRBond2



# Test: time evolution
# d = 2
# krausDim = 1;
# bondDim = 1;
# bondDimTrunc = 2;
# krausDimTrunc = 2;

# GAMMA = 1.0;
# OMEGA = 6.0;
# dt = 0.5;

# sigmaX = 0.5 * [0 +1; +1 0];
# numberOp = [0 0; 0 1];
# hamOp = OMEGA * (kron(sigmaX, numberOp) + kron(numberOp, sigmaX));

# numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);
# hamOp = TensorMap(hamOp, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
# hamDyn = expHam(OMEGA, dt / 2);
# dissDyn = expDiss(GAMMA, dt);

# basis0 = [1, 0];
# basis1 = [0, 1];
# tensorBase0 = zeros(ComplexF64, 1, krausDim, d, 1);
# tensorBase0[:, :, 1, 1] = reshape([basis0[1]], 1, 1);
# tensorBase0[:, :, 2, 1] = reshape([basis0[2]], 1, 1);
# tensorBase1 = zeros(ComplexF64, 1, krausDim, d, 1);
# tensorBase1[:, :, 1, 1] = reshape([basis1[1]], 1, 1);
# tensorBase1[:, :, 2, 1] = reshape([basis1[2]], 1, 1);

# Go = TensorMap(
#     tensorBase1,
#     ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim),
#     ComplexSpace(d) ⊗ ComplexSpace(bondDim),
# );
# Ge = TensorMap(
#     tensorBase0,
#     ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim),
#     ComplexSpace(d) ⊗ ComplexSpace(bondDim),
# );
# Lo = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));
# Le = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));

# # left-isometry (odd bond)
# @tensor leftEnv[-2; -1] :=
#     Go[1, 2, 3, -1] * Le[4, 1] * conj(Le[4, 5]) * conj(Go[5, 2, 3, -2]);
# @show leftEnv
# leftEnv_matrix = reshape(
#     convert(Array, leftEnv), dim(space(leftEnv, 1)), dim(space(leftEnv, 1))
# );
# @assert isapprox(leftEnv_matrix, I)

# # left-isometry (even bond)
# @tensor leftEnv[-2; -1] :=
#     Ge[1, 2, 3, -1] * Lo[4, 1] * conj(Lo[4, 5]) * conj(Ge[5, 2, 3, -2]);
# @show leftEnv
# leftEnv_matrix = reshape(
#     convert(Array, leftEnv), dim(space(leftEnv, 1)), dim(space(leftEnv, 1))
# );
# @assert isapprox(leftEnv_matrix, I)

# # right-isometry (odd bond)
# @tensor rightEnv[-1; -2] :=
#     Ge[-1, 1, 2, 3] * Le[3, 4] * conj(Le[5, 4]) * conj(Ge[-2, 1, 2, 5]);
# @show rightEnv
# rightEnv_matrix = reshape(
#     convert(Array, rightEnv), dim(space(rightEnv, 1)), dim(space(rightEnv, 1))
# );
# @assert isapprox(rightEnv_matrix, I)

# # right-isometry (even bond)
# @tensor rightEnv[-1; -2] :=
#     Go[-1, 1, 2, 3] * Lo[3, 4] * conj(Lo[5, 4]) * conj(Go[-2, 1, 2, 5]);
# @show rightEnv
# rightEnv_matrix = reshape(
#     convert(Array, rightEnv), dim(space(rightEnv, 1)), dim(space(rightEnv, 1))
# );
# @assert isapprox(rightEnv_matrix, I)

# @show computeSiteExpVal!(Go, Ge, Lo, Le, numberOp)
# Go, Ge, Lo, Le = iTEBD_for_LPTN!(
#     Go, Ge, Lo, Le, hamDyn, hamDyn, dissDyn, bondDimTrunc, krausDimTrunc
# );

# println("After one time step evolution")
# # left-isometry (odd bond)
# @tensor leftEnv[-2; -1] :=
#     Go[1, 2, 3, -1] * Le[4, 1] * conj(Le[4, 5]) * conj(Go[5, 2, 3, -2]);
# @show leftEnv
# leftEnv_matrix = reshape(
#     convert(Array, leftEnv), dim(space(leftEnv, 1)), dim(space(leftEnv, 1))
# );
# @assert isapprox(leftEnv_matrix, I)

# # # left-isometry (even bond)
# # @tensor leftEnv[-2; -1] := Ge[1, 2, 3, -1] * Lo[4, 1] * conj(Lo[4, 5]) * conj(Ge[5, 2, 3, -2]);
# # @show leftEnv
# # leftEnv_matrix = reshape(convert(Array, leftEnv), dim(space(leftEnv, 1)), dim(space(leftEnv, 1)));
# # @assert isapprox(leftEnv_matrix, I)

# # right-isometry (odd bond)
# @tensor rightEnv[-1; -2] :=
#     Ge[-1, 1, 2, 3] * Le[3, 4] * conj(Le[5, 4]) * conj(Ge[-2, 1, 2, 5]);
# @show rightEnv
# rightEnv_matrix = reshape(
#     convert(Array, rightEnv), dim(space(rightEnv, 1)), dim(space(rightEnv, 1))
# );
# @assert isapprox(rightEnv_matrix, I)

# # # right-isometry (even bond)
# # @tensor rightEnv[-1; -2] := Go[-1, 1, 2, 3] * Lo[3, 4] * conj(Lo[5, 4]) * conj(Go[-2, 1, 2, 5]);
# # @show rightEnv
# # rightEnv_matrix = reshape(convert(Array, rightEnv), dim(space(rightEnv, 1)), dim(space(rightEnv, 1)));
# # @assert isapprox(rightEnv_matrix, I)

# # bring back to 
# @show computeSiteExpVal!(Go, Ge, Lo, Le, numberOp)

nothing
