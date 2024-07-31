using TensorKit
using KrylovKit # Lanczos - real EVal, Oddrnoldi - complex Eval

function applyGate!(leftT, rightT, weightMid, weightSide, op, bondDim, truncErr)
    @tensor bondTensor[-1 -2 -3; -4] :=
        weightSide[-1, 1] *
        leftT[1, -2, 2] *
        weightMid[2, 3] *
        rightT[3, -3, 4] *
        weightSide[4, -4]
    @tensor bondTensor[-1 -2 -3; -4] := op[-2, -3, 1, 2] * bondTensor[-1, 1, 2, -4]
    bondTensor /= norm(bondTensor)

    U, S, V, ϵ = tsvd(
        bondTensor,
        (1, 2),
        (3, 4);
        trunc=truncdim(bondDim) & truncerr(truncErr),
        alg=TensorKit.SVD(),
    )

    @tensor leftT[-1 -2; -3] := pinv(weightSide)[-1, 1] * U[1, -2, -3]
    @tensor rightT[-1 -2; -3] := V[-1, -2, 1] * pinv(weightSide)[1, -3]
    weightMid = S / norm(S)

    return leftT, rightT, weightMid
end

function applyUGateLPTN!(leftT, rightT, weightMid, weightSide, op, bondDim, truncErr)
    QR, R = leftorth(leftT, (1, 2), (3, 4); alg=QRpos())
    QR = permute(QR, (1, 2), (3,))
    R = permute(R, (1,), (2, 3))

    L, QL = rightorth(rightT, (1, 3), (2, 4); alg=LQpos())
    L = permute(L, (1,), (2, 3))
    QL = permute(QL, (1, 2), (3,))

    @tensor bondTensor[-1 -2 -3; -4 -5 -6] :=
        weightSide[-1, 1] *
        QR[1, -2, 2] *
        R[2, 3, 4] *
        weightMid[4, 5] *
        L[5, 6, 7] *
        QL[7, -3, 8] *
        op[3, 6, -4, -5] *
        weightSide[8, -6]

    U, S, V, ϵ = tsvd(
        bondTensor,
        (1, 2, 4),
        (3, 5, 6);
        trunc=truncdim(bondDim) & truncerr(truncErr),
        alg=TensorKit.SVD(),
    )
    weightMid = S / norm(S) # to normalise truncated bondTensor

    @tensor leftT[-1 -2; -3 -4] := pinv(weightSide)[-1, 1] * U[1, -2, -3, -4]
    @tensor rightT[-1 -2; -3 -4] := V[-1, -2, -3, 1] * pinv(weightSide)[1, -4]

    return leftT, rightT, weightMid, ϵ
end

function applyDGateLPTN!(leftT, rightT, weightSide, krausOp, krausDim, truncErr)
    @tensor leftT[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * leftT[-1, -3, 1, -5]
    leftT /= norm(leftT)

    U, S, V, ϵ = tsvd(
        leftT,
        (2, 3),
        (1, 4, 5);
        trunc=truncdim(krausDim) & truncerr(truncErr),
        alg=TensorKit.SVD(),
    )
    S /= norm(S) # normalise truncated bondTensor
    leftT = permute(S * V, (2, 1), (3, 4))

    @tensor rightT[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * rightT[-1, -3, 1, -5]
    U, S, V, ϵ = tsvd(
        rightT,
        (2, 3),
        (1, 4, 5);
        trunc=truncdim(krausDim) & truncerr(truncErr),
        alg=TensorKit.SVD(),
    )
    S /= norm(S) # normalise truncated bondTensor
    rightT = permute(S * V, (2, 1), (3, 4))

    # orthonormalize
    @tensor bondTensor[-1 -2 -3; -4 -5 -6] :=
        weightSide[-1, 1] * leftT[1, -2, -4, 2] * rightT[2, -3, -5, 3] * weightSide[3, -6]
    U, S, V, _ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6))

    weightMid = S / norm(S)
    @tensor leftT[-1 -2; -3 -4] := pinv(weightSide)[-1, 1] * U[1, -2, -3, -4]
    @tensor rightT[-1 -2; -3 -4] := V[-1, -2, -3, 1] * pinv(weightSide)[1, -4]

    return leftT, rightT, weightMid
end

function computeBondEnergy(Go, Ge, Lo, Le, op)
    """
    Compute bond energy for iMPS Ansatz
    """

    # energy for odd bond
    @tensor bondTensorO[-1 -2 -3; -4] :=
        Le[-1, 1] * Go[1, -2, 2] * Lo[2, 3] * Ge[3, -3, 4] * Le[4, -4]
    @tensor energyO =
        bondTensorO[1, 2, 3, 4] * op[5, 6, 2, 3] * conj(bondTensorO[1, 5, 6, 4])
    energyO /= norm(bondTensorO)

    # energy for even bond
    @tensor bondTensorE[-1 -2 -3; -4] :=
        Lo[-1, 1] * Ge[1, -2, 2] * Le[2, 3] * Go[3, -3, 4] * Lo[4, -4]
    @tensor energyE =
        bondTensorE[1, 2, 3, 4] * op[5, 6, 2, 3] * conj(bondTensorE[1, 5, 6, 4])
    energyE /= norm(bondTensorE)

    if abs(imag(energyO)) < 1e-12 && abs(imag(energyE)) < 1e-12
        return (1 / 2) * (energyO + energyE)
    else
        ErrorException("Oops! Complex energy is found.")
    end
end

function computeBondEnergyH(Go, Ge, Lo, Le, op)
    """
    Compute bond energy for iLPTN Ansatz
    """
    # energy for odd bond
    @tensor bondTensorO[-1 -2 -3; -4 -5 -6] :=
        Le[-1, 1] * Go[1, -2, -4, 2] * Lo[2, 3] * Ge[3, -3, -5, 4] * Le[4, -6]
    @tensor energyO =
        bondTensorO[5, 6, 7, 3, 4, 8] * op[3, 4, 1, 2] * conj(bondTensorO[5, 6, 7, 1, 2, 8])
    energyO /= norm(bondTensorO)

    # energy for even bond
    @tensor bondTensorE[-1 -2 -3; -4 -5 -6] :=
        Lo[-1, 1] * Ge[1, -2, -4, 2] * Le[2, 3] * Go[3, -3, -5, 4] * Lo[4, -6]
    @tensor energyE =
        bondTensorE[5, 6, 7, 3, 4, 8] * op[3, 4, 1, 2] * conj(bondTensorE[5, 6, 7, 1, 2, 8])
    energyE /= norm(bondTensorE)

    if abs(imag(energyO)) < 1e-12 && abs(imag(energyE)) < 1e-12
        return (1 / 2) * (energyO + energyE)
    else
        ErrorException("Oops! Complex energy is found.")
    end
end

function computeBondEnergyD(Go, Ge, Lo, Le, krausOp)

    # energy for odd site -> bondTensor = Le - Go - Lo
    @tensor bondTensorO[-1 -2; -3 -4] := Le[-1, 1] * Go[1, -2, -3, 2] * Lo[2, -4]
    @tensor bondTensorOK[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * bondTensorO[-1, -3, 1, -5]
    # iDT = TensorMap(ones, space(bondTensorO, 3), space(bondTensorOK, 2) ⊗ space(bondTensorOK, 3));

    fuser = isometry(space(bondTensorO, 2), space(bondTensorOK, 2) ⊗ space(bondTensorOK, 3))

    @tensor energyO =
        bondTensorOK[1, 2, 3, 4, 5] * fuser[6, 2, 3] * conj(bondTensorO[1, 6, 4, 5])
    energyO /= norm(bondTensorO)

    # # energy for even bond -> bondTensor = Lo - Ge - Le
    # @tensor bondTensorE[-1 -2 -3; -4 -5 -6] := Lo[-1, 1] * Ge[1, -2, -4, 2] * Le[2, 3] * Go[3, -3, -5, 4] * Lo[4, -6];
    # @tensor energyE = bondTensorE[5, 6, 7, 3, 4, 8] * op[3, 4, 1, 2] * conj(bondTensorE[5, 6, 7, 1, 2, 8]);
    # energyE /= norm(bondTensorE);

    if abs(imag(energyO)) < 1e-12
        return (1 / 2) * (energyO)
    else
        ErrorException("Oops! Complex energy is found.")
    end
end

function iTEBD!(Go, Ge, Lo, Le, expHo, expHe, H, bondDim; truncErr=1e-6)
    """
    2nd order TEBD for unitary evolution for one time step
    """

    # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    Go, Ge, Lo = applyGate!(Go, Ge, Lo, Le, expHo, bondDim, truncErr)

    # even bond update -> bondTensor = Lo - Ge - Le - Go - Lo
    Ge, Go, Le = applyGate!(Ge, Go, Le, Lo, expHe, bondDim, truncErr)

    # # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    # Go, Ge, Lo = applyGate!(Go, Ge, Lo, Le, expHo, bondDim, truncErr);

    return Go, Ge, Lo, Le, computeBondEnergy(Go, Ge, Lo, Le, H)
end

function iTEBD_for_LPTN!(
    Go, Ge, Lo, Le, expHo, expHe, expD, bondDim, krausDim; truncErr=1e-6
)
    ### XXX: How to calculate energy
    """
    2nd order iTEBD with dissipative layer for one time step

    Params:
    - Go, Ge, Lo, Le: from 2-site canonical form

    """

    # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    Go, Ge, Lo = applyUGateLPTN!(Go, Ge, Lo, Le, expHo, bondDim, truncErr)

    # even bond update -> bondTensor = Lo - Ge - Le - Go - Lo
    Ge, Go, Le = applyUGateLPTN!(Ge, Go, Le, Lo, expHe, bondDim, truncErr)

    # dissipative process on bondTensor = Lo - Ge - Le - Go - Lo
    Ge, Go, Le = applyDGateLPTN!(Ge, Go, Lo, expD, krausDim, truncErr)

    # # left-isometry
    # @tensor leftEnv[-2; -1] := Lo[1, 2] * Ge[2, 3, 4, -1] * conj(Lo[1, 5]) * conj(Ge[5, 3, 4, -2]);
    # @show space(leftEnv)
    # @show leftEnv

    # # right-isometry
    # @tensor rightEnv[-1; -2] := Go[-1, 1, 2, 3] * Lo[3, 4] * conj(Lo[5, 4]) * conj(Go[-2, 1, 2, 5]);
    # @show space(rightEnv)
    # @show rightEnv

    # even bond update -> bondTensor = Lo - Ge - Le - Go - Lo
    Ge, Go, Le = applyUGateLPTN!(Ge, Go, Le, Lo, expHe, bondDim, truncErr)

    # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    Go, Ge, Lo = applyUGateLPTN!(Go, Ge, Lo, Le, expHo, bondDim, truncErr)

    return Go, Ge, Lo, Le
    #    computeBondEnergy(Go, Ge, Lo, Le, H) + computeBondEnergyD(Go, Ge, Lo, Le, krausOp)        

end
