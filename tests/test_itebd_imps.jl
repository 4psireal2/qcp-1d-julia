using LinearAlgebra
using TensorKit
using KrylovKit

include("../src/itebd.jl")
include("../src/imps.jl")

d = 2;
bondDim = 3;
bondDimTrunc = 5;
maxiter = 1000;
tol = 1e-6;

Go = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Ge = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
Lo = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));
Le = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));

# Test: canonical form
# bond1 = odd bond - bondTensor = Le - Go - Lo - Ge - Le
# bond2 = even bond - bondTensor = Lo - Ge - Le - Go - Lo

initGuess = Matrix(I, bondDim, bondDim) ./ bondDim;
transferOpLBond1 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim));
transferOpRBond2 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim));

transferOpLBond1, transferOpLBond2 = leftContraction!(transferOpLBond1, Le, Go, Lo, Ge)
transferOpRBond2, transferOpRBond1 = rightContraction!(transferOpRBond2, Le, Go, Lo, Ge)

Ge, Le, Go, gaugeLEven, gaugeREven = orthonormalizeiMPS(
    transferOpLBond1, transferOpRBond1, Lo, Ge, Le, Go
)
Go, Lo, Ge, gaugeLOdd, gaugeROdd = orthonormalizeiMPS(
    transferOpLBond2, transferOpRBond2, Le, Go, Lo, Ge
)

@tensor transferOpLBond1[-1; -2] :=
    gaugeLEven'[-1, 1] * transferOpLBond1[1, 2] * gaugeLEven[2, -2]
@show transferOpLBond1

@tensor transferOpRBond1[-1; -2] :=
    gaugeREven[-1, 1] * transferOpRBond1[1, 2] * gaugeREven'[2, -2]
@show transferOpRBond1

@tensor transferOpLBond2[-1; -2] :=
    gaugeLOdd'[-1, 1] * transferOpLBond2[1, 2] * gaugeLOdd[2, -2]
@show transferOpLBond2

@tensor transferOpRBond2[-1; -2] :=
    gaugeROdd[-1, 1] * transferOpRBond2[1, 2] * gaugeROdd'[2, -2]
@show transferOpRBond2

@show computeCorrLen(Go, Ge, Lo, Le)

# Test: iTEBD - TFI model
# Ref: [https://tenpy.readthedocs.io/en/latest/toycodes/solution_3_dmrg.html#Infinite-DMRG]
delta = 0.01;
nTimeSteps = 1000;
J = 1.0;
g = 0.5;
unitCellSize = 2;

Sx = TensorMap([0 1; 1 0], ℂ^2, ℂ^2);
Sz = TensorMap([1 0; 0 -1], ℂ^2, ℂ^2);
Id = TensorMap([1 0; 0 1], ℂ^2, ℂ^2);
H = -J * (Sz ⊗ Sz) + g * (Sx ⊗ Id);

expHo = exp(-delta * H);
expHe = exp(-delta * H);

# initial state of even and odd sites - maximally entangled state?
Go = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin up
Ge = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1); # spin down
Lo = TensorMap(ones, ℂ^1, ℂ^1);
Le = TensorMap(ones, ℂ^1, ℂ^1);

# left-isometry (even bond)
@tensor leftEnv[-2; -1] := Le[1, 2] * Go[2, 3, -1] * conj(Le[1, 4]) * conj(Go[4, 3, -2]);
@show leftEnv

# left-isometry (odd bond)
@tensor leftEnv[-2; -1] := Lo[1, 2] * Ge[2, 3, -1] * conj(Lo[1, 4]) * conj(Ge[4, 3, -2]);
@show leftEnv

# right-isometry (even bond)
@tensor rightEnv[-1; -2] := Ge[-1, 1, 2] * Le[2, 3] * conj(Le[4, 3]) * conj(Ge[-2, 1, 4]);
@show rightEnv

# right-isometry (odd bond)
@tensor rightEnv[-1; -2] := Go[-1, 1, 2] * Lo[2, 3] * conj(Lo[4, 3]) * conj(Go[-2, 1, 4]);
@show rightEnv

# full time evolution
let
    Go = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1) # spin up
    Ge = TensorMap([1, 0], ℂ^1 ⊗ ℂ^2, ℂ^1) # spin down
    Lo = TensorMap(ones, ℂ^1, ℂ^1)
    Le = TensorMap(ones, ℂ^1, ℂ^1)

    for i in 1:nTimeSteps
        Go, Ge, Lo, Le = iTEBD!(Go, Ge, Lo, Le, expHo, expHe, 10)

        if mod(i, 100) == 0
            @show compute2SiteExpVal(Go, Ge, Lo, Le, H)
            @show compute1SiteExpVal(Go, Ge, Lo, Le, Sz)
        end
    end
end

# correlation length for 2-site unit cell

println("E_exact =  -1.063544409973372")
println("m_lit   =  0.9646523684425512")

println("Check single-site canonical form after full time evolution")

# left-isometry (even bond)
@tensor leftEnv[-2; -1] := Le[1, 2] * Go[2, 3, -1] * conj(Le[1, 4]) * conj(Go[4, 3, -2]);
@show leftEnv

# left-isometry (odd bond)
@tensor leftEnv[-2; -1] := Lo[1, 2] * Ge[2, 3, -1] * conj(Lo[1, 4]) * conj(Ge[4, 3, -2]);
@show leftEnv

# right-isometry (even bond)
@tensor rightEnv[-1; -2] := Ge[-1, 1, 2] * Le[2, 3] * conj(Le[4, 3]) * conj(Ge[-2, 1, 4]);
@show rightEnv

# right-isometry (odd bond)
@tensor rightEnv[-1; -2] := Go[-1, 1, 2] * Lo[2, 3] * conj(Lo[4, 3]) * conj(Go[-2, 1, 4]);
@show rightEnv

### WARNING: Very unstable !!!
# full time evolution with random initial state
# let
#     println("Time evolution with random initial state")
#     bondDim = 1
#     Go = TensorMap(
#         [0.5, 0.5], ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim)
#     )
#     Ge = TensorMap(
#         [0.5, 0.5], ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim)
#     )
#     Lo = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim))
#     Le = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim))
#     initGuess = Matrix(I, bondDim, bondDim) ./ bondDim
#     transferOpLBond1 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim))
#     transferOpRBond2 = TensorMap(initGuess, ComplexSpace(bondDim), ComplexSpace(bondDim))

#     transferOpLBond1, transferOpLBond2 = leftContraction!(transferOpLBond1, Le, Go, Lo, Ge)
#     transferOpRBond2, transferOpRBond1 = rightContraction!(transferOpRBond2, Le, Go, Lo, Ge)
#     Ge, Le, Go, gaugeLEven, gaugeREven = orthonormalizeiMPS(
#         transferOpLBond1, transferOpRBond1, Lo, Ge, Le, Go
#     )
#     Go, Lo, Ge, gaugeLOdd, gaugeROdd = orthonormalizeiMPS(
#         transferOpLBond2, transferOpRBond2, Le, Go, Lo, Ge
#     )

#     for i in 1:50
#         Go, Ge, Lo, Le = iTEBD!(Go, Ge, Lo, Le, expHo, expHe, 10)

#         if mod(i, 1) == 0
#             @show i
#             initGuess = Matrix(I, dim(space(Lo, 1)), dim(space(Lo, 1))) ./ dim(space(Lo, 1))
#             transferOpLBond1 = TensorMap(initGuess, space(Lo, 1), space(Lo, 1))

#             initGuess = Matrix(I, dim(space(Lo, 1)), dim(space(Lo, 1))) ./ dim(space(Lo, 1))
#             transferOpRBond2 = TensorMap(initGuess, space(Lo, 1), space(Lo, 1))

#             transferOpLBond1, transferOpLBond2 = leftContraction!(
#                 transferOpLBond1, Le, Go, Lo, Ge
#             )
#             transferOpRBond2, transferOpRBond1 = rightContraction!(
#                 transferOpRBond2, Le, Go, Lo, Ge
#             )

#             Ge, Le, Go, gaugeLEven, gaugeREven = orthonormalizeiMPS(
#                 transferOpLBond1, transferOpRBond1, Lo, Ge, Le, Go
#             )
#             Go, Lo, Ge, gaugeLOdd, gaugeROdd = orthonormalizeiMPS(
#                 transferOpLBond2, transferOpRBond2, Le, Go, Lo, Ge
#             )

#             @show compute2SiteExpVal(Go, Ge, Lo, Le, H)
#             @show compute1SiteExpVal(Go, Ge, Lo, Le, Sz)
#         end
#     end
# end

# println("E_exact =  -1.063544409973372")
# println("m_lit   =  0.9646523684425512")

nothing
