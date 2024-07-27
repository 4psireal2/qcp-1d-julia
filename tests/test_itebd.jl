using LinearAlgebra
using TensorKit

include("../src/itebd.jl")


d = 2; 
bondDim = 3;
maxiter = 1000;
tol=1e-12;


Go = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Ge = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Lo = TensorMap(randn, ComplexSpace(bondDim), ComplexSpace(bondDim));
Le = TensorMap(randn, ComplexSpace(bondDim), ComplexSpace(bondDim));

# Test: canonical form
## across even-odd link
### coarse grain Ge - Le - Go -> bondTensor
@tensor bondTensor[-1 -2 -3; -4] := Ge[-1, -2, 1] * Le[1, 2] * Go[2, -3, -4];

@tensor rightEnv[-1 -4; -2 -3] := bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 4]) * Lo[3, -3] * conj(Lo[4, -4]);
@tensor transferR[-1; -2] := bondTensor[-1, 1, 2, 3] * conj(bondTensor[-2, 1, 2, 5]) * Lo[3, 4] * conj(Lo[5, 4]);

@tensor leftEnv[-1 -4; -2 -3] := Lo[-1, 1] * bondTensor[1, 2, 3, -3] * conj(Lo[-2, 4]) * conj(bondTensor[4, 2, 3, -4]);
@tensor transferL[-2; -1] := Lo[1, 2] * bondTensor[2, 3, 4, -1] * conj(Lo[1, 5]) * conj(bondTensor[5, 3, 4, -2]);

println("Initial guesses for transfer operators")
@show transferL
@show transferR

println("transfer operators as dominant eigenvectors")
transferL = findTransferOp(transferL, leftEnv);
@show transferL

transferR = findTransferOp(transferR, rightEnv);
@show transferR

Ge_can, Go_can, Le_can, Lo_can = orthonormalizeiMPS!(bondTensor, Le, Lo, transferL, transferR);

## left-isometry
@tensor leftEnv[-2; -1] := Ge_can[1, 2, -1] * Lo_can[3, 1] * conj(Lo_can[3, 4]) * conj(Ge_can[4, 2, -2]);
@show leftEnv
leftEnv_matrix = reshape(convert(Array, leftEnv), dim(space(leftEnv)[1]), dim(space(leftEnv)[1]));
@assert isapprox(leftEnv_matrix, I)

## right-isometry
@tensor rightEnv[-1; -2] := Go_can[-1, 1, 2] * Lo_can[2, 3] * conj(Lo_can[4, 3]) * conj(Go_can[-2, 1, 4])
rightEnv_matrix = reshape(convert(Array, rightEnv), dim(space(rightEnv)[1]), dim(space(rightEnv)[1]));
@show rightEnv
@assert isapprox(rightEnv_matrix, I)


# across odd-even link


nothing

