using TensorKit
using KrylovKit




function updateEnvL(
    EnvL::TensorMap, mpsT::TensorMap, op::TensorMap, mpsB::TensorMap
)::TensorMap
    @tensor newEnvL[-1; -2 -3] :=
        EnvL[1, 5, 3] * mpsT[5, 4, -2] * op[3, 2, 4, -3] * conj(mpsB[1, 2, -1])

    return newEnvL
end

function updateEnvR(
    EnvR::TensorMap, mpsT::TensorMap, op::TensorMap, mpsB::TensorMap
)::TensorMap
    @tensor newEnvR[-1 -2; -3] :=
        mpsT[-1, 1, 2] * op[-2, 4, 1, 3] * conj(mpsB[-3, 4, 5]) * EnvR[2, 3, 5]

    return newEnvR
end


function initializeEnvs(mps, op; centerPos::Int64=1)
    """
    Create left and right environments for DMRG2
    """
    length(mps) == length(op) || throw(ArgumentError("dimension mismatch"))

    N = length(mps)

    envL = Vector{TensorMap}(undef, N)
    envR = Vector{TensorMap}(undef, N)
    envL[1] = TensorMap(ones, space(mps[1], 1), space(mps[1], 1) ⊗ space(op[1], 1))
    envR[N] = TensorMap(ones, space(mps[N], 3)' ⊗ space(op[N], 4)', space(mps[N], 3)')
    # compute envL up to (centerPos - 1)
    for i in 1:1:(centerPos - 1)
        envL[i + 1] = updateEnvL(envL[i], mps[i], op[i], mps[i])
    end

    # compute envR up to (centerPos + 1)
    for i in N:-1:(centerPos + 1)
        envR[i - 1] = updateEnvR(envR[i], mps[i], op[i], mps[i])
    end

    return envL, envR
end


function applyH1(X::TensorMap, EnvL::TensorMap, op::TensorMap, EnvR::TensorMap)::TensorMap
    @tensor X[-1 -2; -3] := EnvL[-1, 1, 4] * X[1, 2, 3] * op[4, -2, 2, 5] * EnvR[3, 5, -3]

    return X
end


function findHSum(localH, N)

    hEnd = localH
    for i = 1 : (N-2)
        @tensor hEnd[-1 -2 -3 -4; -5 -6 -7 -8] := hEnd[-1, -2, 1, -5, -6, 2] * localH[2, -3, -4, 1, -7, -8]
        fuserOut = isometry(fuse(space(hEnd, 2), space(hEnd, 3)), space(hEnd, 2) * space(hEnd, 3))
        fuserIn = isometry(space(hEnd, 6)' * space(hEnd,7)', fuse(space(hEnd, 6), space(hEnd,7)))


        @tensor hEnd[-1 -2 -3; -4 -5 -6] := hEnd[-1, 1, 2, -3, -4, 3, 4, -6] * fuserOut[-2, 1, 2] * fuserIn[3, 4, -5]
    end

    fuserFinIn = isometry(space(hEnd, 4)' * space(hEnd,5)', fuse(space(hEnd, 4), space(hEnd,5)))
    fuserFinOut = isometry(fuse(space(hEnd, 2), space(hEnd, 3)), space(hEnd, 2) * space(hEnd, 3))
    @tensor hEnd[-1 -2; -3 -4] := hEnd[-1, 1, 2, 3, 4, -4] * fuserFinOut[-2, 1, 2] * fuserFinIn[3, 4, -3]


    return hEnd

end


# Test for TFI model
d = 2
N = 5
J = 1.0
g = 1.0

Id = [+1 0; 0 +1]
Sx = [0 +1; +1 0]
Sz = [+1 0; 0 -1]

H = -J * kron(Sz, Sz) - g * 0.5 * (kron(Sx, Id) + kron(Id, Sx));
H = TensorMap(H, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1))



hSum = findHSum(H, N)

testState = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d^N), ComplexSpace(1))
testState /norm(testState)

envL = TensorMap(ones, space(testState, 1), space(testState, 1) ⊗ space(hSum, 1))
envR = TensorMap(ones, space(testState, 3)' ⊗ space(hSum, 4)', space(testState, 3)')

eigenVal, eigenVec =
eigsolve(testState, 1, :SR, Lanczos(; tol=1e-16, maxiter=200)) do x
    applyH1(x, envL, hSum, envR)
end



nothing
