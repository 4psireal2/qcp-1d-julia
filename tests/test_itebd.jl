using LinearAlgebra
using TensorKit

include("../src/itebd.jl")
include("../src/imps.jl")

d = 2;
bondDim = 3;
bondDimTrunc = 5;
maxiter = 1000;
tol = 1e-10;

Go = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Ge = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Lo = TensorMap(randn, ComplexSpace(bondDim), ComplexSpace(bondDim));
Le = TensorMap(randn, ComplexSpace(bondDim), ComplexSpace(bondDim));

# Test: canonical form
## across even-odd link
### coarse grain Ge - Le - Go -> bondTensor
@tensor bondTensor[-1 -2 -3; -4] := Ge[-1, -2, 1] * Le[1, 2] * Go[2, -3, -4];

@tensor rightEnv[-1 -4; -2 -3] :=
    bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 4]) * Lo[3, -3] * conj(Lo[4, -4]);
@tensor transferR[-1; -2] :=
    bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 5]) * Lo[3, 4] * conj(Lo[5, 4]);

@tensor leftEnv[-1 -4; -2 -3] :=
    Lo[-1, 1] * bondTensor[1, 2, 3, -3] * conj(Lo[-2, 4]) * conj(bondTensor[4, 2, 3, -4]);
@tensor transferL[-2; -1] :=
    Lo[1, 2] * bondTensor[2, 3, 4, -1] * conj(Lo[1, 5]) * conj(bondTensor[5, 3, 4, -2]);

println("Initial guesses for transfer operators")
@show transferL
@show transferR

println("transfer operators as dominant eigenvectors")
transferL = findTransferOp(transferL, leftEnv);
@show transferL

transferR = findTransferOp(transferR, rightEnv);
@show transferR

Ge_can, Go_can, Le_can, Lo_can = orthogonalizeiMPS!(bondTensor, Lo, transferL, transferR);

### left-isometry (even bond)
@tensor leftEnv[-2; -1] :=
    Ge_can[1, 2, -1] * Lo_can[3, 1] * conj(Lo_can[3, 4]) * conj(Ge_can[4, 2, -2]);
@show leftEnv
leftEnv_matrix = reshape(
    convert(Array, leftEnv), dim(space(leftEnv)[1]), dim(space(leftEnv)[1])
);
@assert isapprox(leftEnv_matrix, I)

# ### left-isometry (odd bond)
# @tensor leftEnv[-2; -1] := Go_can[1, 2, -1] * Le_can[3, 1] * conj(Le_can[3, 4]) * conj(Go_can[4, 2, -2]);
# @show leftEnv
# leftEnv_matrix = reshape(convert(Array, leftEnv), dim(space(leftEnv)[1]), dim(space(leftEnv)[1]));
# @assert isapprox(leftEnv_matrix, I)

### right-isometry (even bond)
@tensor rightEnv[-1; -2] :=
    Go_can[-1, 1, 2] * Lo_can[2, 3] * conj(Lo_can[4, 3]) * conj(Go_can[-2, 1, 4])
rightEnv_matrix = reshape(
    convert(Array, rightEnv), dim(space(rightEnv)[1]), dim(space(rightEnv)[1])
);
@show rightEnv
@assert isapprox(rightEnv_matrix, I)

# ### right-isometry (odd bond)
# @tensor rightEnv[-1; -2] := Ge_can[-1, 1, 2] * Le_can[2, 3] * conj(Le_can[4, 3]) * conj(Ge_can[-2, 1, 4])
# rightEnv_matrix = reshape(convert(Array, rightEnv), dim(space(rightEnv)[1]), dim(space(rightEnv)[1]));
# @show rightEnv
# @assert isapprox(rightEnv_matrix, I)

@tensor bondTensor[-1 -2 -3; -4] := Go[-1, -2, 1] * Lo[1, 2] * Ge[2, -3, -4];
@tensor rightEnv[-1 -4; -2 -3] :=
    bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 4]) * Le[3, -3] * conj(Le[4, -4]);
@tensor transferR[-1; -2] :=
    bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 5]) * Le[3, 4] * conj(Le[5, 4]);

@tensor leftEnv[-1 -4; -2 -3] :=
    Le[-1, 1] * bondTensor[1, 2, 3, -3] * conj(Le[-2, 4]) * conj(bondTensor[4, 2, 3, -4]);
@tensor transferL[-2; -1] :=
    Le[1, 2] * bondTensor[2, 3, 4, -1] * conj(Le[1, 5]) * conj(bondTensor[5, 3, 4, -2]);

transferL = findTransferOp(transferL, leftEnv);
transferR = findTransferOp(transferR, rightEnv);

Go_can, Ge_can, Lo_can, Le_can = orthogonalizeiMPS!(bondTensor, Le, transferL, transferR);

### left-isometry (odd bond)
@tensor leftEnv[-2; -1] :=
    Go_can[1, 2, -1] * Le_can[3, 1] * conj(Le_can[3, 4]) * conj(Go_can[4, 2, -2]);
@show leftEnv
leftEnv_matrix = reshape(
    convert(Array, leftEnv), dim(space(leftEnv)[1]), dim(space(leftEnv)[1])
);
@assert isapprox(leftEnv_matrix, I)

### left-isometry (even bond)
@tensor leftEnv[-2; -1] :=
    Ge_can[1, 2, -1] * Lo_can[3, 1] * conj(Lo_can[3, 4]) * conj(Ge_can[4, 2, -2]);
@show leftEnv
leftEnv_matrix = reshape(
    convert(Array, leftEnv), dim(space(leftEnv)[1]), dim(space(leftEnv)[1])
);
@assert isapprox(leftEnv_matrix, I)

# Test: iTEBD - TFI model
# Ref: [https://tenpy.readthedocs.io/en/latest/toycodes/solution_3_dmrg.html#Infinite-DMRG]
# delta = 0.01; 
# nTimeSteps = 1000;
# J = 1.0;
# g = 0.5;
# unitCellSize = 2;

# Sx = TensorMap([0 1; 1 0],  ℂ^2,  ℂ^2);
# Sz = TensorMap([1 0; 0 -1],  ℂ^2,  ℂ^2);
# Id = TensorMap([1 0; 0 1],  ℂ^2,  ℂ^2);
# H = -J * (Sz ⊗ Sz) + g * (Sx ⊗ Id);

# expHo = exp(- delta * H);
# expHe = exp(- delta * H);

# H_mat = reshape(convert(Array, expHo), 4, 4);
# @show H_mat

# # initial state of even and odd sites - maximally entangled state?
# Go = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin up
# Ge = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin down
# Lo = TensorMap(ones, ℂ^1, ℂ^1);
# Le = TensorMap(ones, ℂ^1, ℂ^1);

# # left-isometry
# @tensor leftEnv[-2; -1] := Le[1, 2] * Go[2, 3, -1] * conj(Le[1, 4]) * conj(Go[4, 3, -2]);
# @show leftEnv

# @tensor leftEnv[-2; -1] := Lo[1, 2] * Ge[2, 3, -1] * conj(Lo[1, 4]) * conj(Ge[4, 3, -2]);
# @show leftEnv

# # right-isometry
# @tensor rightEnv[-1; -2] := Ge[-1, 1, 2] * Le[2, 3] * conj(Le[4, 3]) * conj(Ge[-2, 1, 4]);
# @show rightEnv

# @tensor rightEnv[-1; -2] := Go[-1, 1, 2] * Lo[2, 3] * conj(Lo[4, 3]) * conj(Go[-2, 1, 4]);
# @show rightEnv

# @show Go
# @show Ge

# Go, Ge, Lo, Le, energy = iTEBD!(Go, Ge, Lo, Le, expHo, expHe, H, bondDimTrunc);

# println("Check canonical form after one time step evolution")
# # left-isometry
# @tensor leftEnv[-2; -1] := Le[1, 2] * Go[2, 3, -1] * conj(Le[1, 4]) * conj(Go[4, 3, -2]);
# @show leftEnv

# @tensor leftEnv[-2; -1] := Lo[1, 2] * Ge[2, 3, -1] * conj(Lo[1, 4]) * conj(Ge[4, 3, -2]);
# @show leftEnv

# # right-isometry
# @tensor rightEnv[-1; -2] := Ge[-1, 1, 2] * Le[2, 3] * conj(Le[4, 3]) * conj(Ge[-2, 1, 4]);
# @show rightEnv

# @tensor rightEnv[-1; -2] := Go[-1, 1, 2] * Lo[2, 3] * conj(Lo[4, 3]) * conj(Go[-2, 1, 4]);
# @show rightEnv

# @show Go
# @show Ge

# # full time evolution
# let
#     Go = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin up
#     Ge = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin down
#     Lo = TensorMap(ones, ℂ^1, ℂ^1);
#     Le = TensorMap(ones, ℂ^1, ℂ^1);

#     for i = 1 : nTimeSteps
#         Go, Ge, Lo, Le, energy = iTEBD!(Go, Ge, Lo, Le, expHo, expHe, H, 10);

#         if mod(i ,100) == 0
#             @show energy
#         end
#     end

# end

# println("E_exact =  -1.063544409973372")

# nothing
