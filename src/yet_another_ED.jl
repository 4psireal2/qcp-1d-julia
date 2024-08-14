using TensorKit
using KrylovKit

function applyH1(X::TensorMap, EnvL::TensorMap, op::TensorMap, EnvR::TensorMap)::TensorMap
    @tensor X[-1 -2; -3] := EnvL[-1, 1, 4] * X[1, 2, 3] * op[4, -2, 2, 5] * EnvR[3, 5, -3]

    return X
end

function findHSum(localH, N)
    hEnd = localH
    for i in 1:(N - 2)
        @tensor hEnd[-1 -2 -3 -4; -5 -6 -7 -8] :=
            hEnd[-1, -2, 1, -5, -6, 2] * localH[2, -3, -4, 1, -7, -8]
        fuserOut = isometry(
            fuse(space(hEnd, 2), space(hEnd, 3)), space(hEnd, 2) * space(hEnd, 3)
        )
        fuserIn = isometry(
            space(hEnd, 6)' * space(hEnd, 7)', fuse(space(hEnd, 6), space(hEnd, 7))
        )

        @tensor hEnd[-1 -2 -3; -4 -5 -6] :=
            hEnd[-1, 1, 2, -3, -4, 3, 4, -6] * fuserOut[-2, 1, 2] * fuserIn[3, 4, -5]
    end

    fuserFinIn = isometry(
        space(hEnd, 4)' * space(hEnd, 5)', fuse(space(hEnd, 4), space(hEnd, 5))
    )
    fuserFinOut = isometry(
        fuse(space(hEnd, 2), space(hEnd, 3)), space(hEnd, 2) * space(hEnd, 3)
    )
    @tensor hEnd[-1 -2; -3 -4] :=
        hEnd[-1, 1, 2, 3, 4, -4] * fuserFinOut[-2, 1, 2] * fuserFinIn[3, 4, -3]

    return hEnd
end

# Test for TFI model
d = 2
N = 18
J = 1.0
g = 1.0

identMat = [+1 0; 0 +1]
Sx = [0 +1; +1 0]
Sy = [0 -1*1im; 1im 0]
Sz = [+1 0; 0 -1]

H = -J * kron(Sz, Sz) - g * 0.5 * (kron(Sx, identMat) + kron(identMat, Sx));
# H = kron(Sx, Sx) + kron(Sy, Sy);

# H = TensorMap(
#     H,
#     ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2),
#     ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1),
# )

# hSum = findHSum(H, N)

# testState = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d^N), ComplexSpace(1))
# testState / norm(testState)

# envL = TensorMap(ones, space(testState, 1), space(testState, 1) ⊗ space(hSum, 1))
# envR = TensorMap(ones, space(testState, 3)' ⊗ space(hSum, 4)', space(testState, 3)')

# eigenVal, eigenVec = eigsolve(testState, 1, :SR, Lanczos(; tol=1e-10, maxiter=200)) do x
#     applyH1(x, envL, hSum, envR)
# end

# H = sum(1:(N-1)) do i
#     localoperators = insert!(fill(Id, N-1), i, H)
#     return reduce(kron, H)
# end
H = sum(1:(N-1)) do i
    localoperators = insert!(fill(Id, N-1), i, H)
    return reduce(kron, localoperators)
end

# N=4

# H_sum = []
# for i = 1 : (N-1)
#     H_i = insert!(fill(Id, N-1), i, H)
#     @show H
#     @show reduce(kron, H)
# end

nothing
