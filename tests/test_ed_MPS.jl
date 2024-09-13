"""
Tested against  TenPy implementation
https://tenpy.readthedocs.io/en/latest/toycode_stubs/tfi_exact.html
"""

using TensorKit
using KrylovKit
using LinearAlgebra

d = 2
N = 3
J = 1.0
g = 1.0

function applyH1(X::TensorMap, op::TensorMap)::TensorMap
    @tensor X[-1 -2; -3] := X[-1, 1, -3] * op[-2, 1]

    return X
end

function findHSum(onSiteOp, interactOp, J, g, N; d=2, PBC=false)
    if N >= 20
        println("Oh no! You are too big for being diagonalized exactly...")
    end

    iD = Matrix(1.0I, d, d)
    identities = fill(iD, N)  # Create a vector filled with the identity matrix `iD` of length `L`
    onSiteOpList = []
    interactOpList = []

    for i in 1:N
        onSiteOps = copy(identities)
        onSiteOps[i] = onSiteOp
        push!(onSiteOpList, reduce(kron, onSiteOps))

        interactOps = copy(identities)
        interactOps[i] = interactOp
        push!(interactOpList, reduce(kron, interactOps))
    end

    H_onSite = zeros(d^N, d^N)
    H_interact = zeros(d^N, d^N)

    for i in 1:(N - 1)
        if i == N - 1 && PBC
            H_interact += interactOpList[i] * interactOpList[1] # last site = first site
        else
            H_interact += interactOpList[i] * interactOpList[i + 1]
        end
    end

    for i in 1:N
        H_onSite += onSiteOpList[i]
    end

    return -J * H_interact - g * H_onSite
end

Sx = [0 1; 1 0];
Sz = [1 0; 0 -1];

hSum = findHSum(Sx, Sz, J, g, N; PBC=false)
hSum = TensorMap(hSum, ComplexSpace(d^N), ComplexSpace(d^N))
testState = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d^N), ComplexSpace(1))
testState / norm(testState)

eigenVal, eigenVec = eigsolve(testState, 1, :SR, Lanczos(; tol=1e-10, maxiter=200)) do x
    applyH1(x, hSum)
end

nothing
