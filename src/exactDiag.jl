using TensorKit
using KrylovKit
using LinearAlgebra
using LinearMaps
using Arpack

d = 2
N = 5
J = 1.0
g = 0.5

function exactDiag(stateIn::TensorMap, op, N; d=2)::TensorMap
    stateOut = TensorMap(zeros, ℂ^1 ⊗ ℂ^(d^N), ℂ^1)

    for i in 1:(N - 1)
        spaceL = 1
        spaceR = d^(N - i - 1)
        if i > 1
            spaceL = d^(i - 1)
        elseif i == N - 1
            spaceR = 1
        end

        stateTemp = reshape(stateIn, spaceL, d^2, spaceR)
        stateTemp = TensorMap(stateTemp, ℂ^(spaceL) ⊗ ℂ^(d^2), ℂ^(spaceR))
        @tensor stateTemp[-1 -2; -3] := stateTemp[-1, 1, -3] * op[-2, 1]
        stateTemp = reshape(permutedims(convert(Array, stateTemp), [2, 1, 3]), d^N)
        stateTemp = TensorMap(stateTemp, ℂ^1 ⊗ ℂ^(d^N), ℂ^1)
        stateOut = stateOut + stateTemp
    end
    stateOut = reshape(convert(Array, stateOut), d^N, 1)
    return stateOut
end

testState = ones(2, 2, 2, 2, 2);
Sx = [0 1; 1 0];
Sz = [1 0; 0 -1];
Id = [1 0; 0 1];
H = -J * kron(Sz, Sz) + g * kron(Sx, Id);
H = reshape(H, 4, 4);
H = TensorMap(H, ℂ^4, ℂ^4);
# testOut = exactDiag(testState, H, N)

# testState = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d^N), ComplexSpace(1));
# energy, state, _ = eigsolve(exactDiag, testState, 1, :SR)
doApplyHamClosed = LinearMap(
    psiIn -> exactDiag(psiIn, H, N),
    d^N;
    ismutating=false,
    issymmetric=true,
    ishermitian=true,
    isposdef=false,
)

diagtime = @elapsed Energy, psi = Arpack.eigs(
    doApplyHamClosed; nev=1, tol=1e-10, which=:SR, maxiter=300
);

nothing
