using TensorKit
using KrylovKit # Lanczos - real EVal, Arnoldi - complex Eval

function computeSiteExpVal!(Go, Ge, Lo, Le, onSiteOp)
    """
    Args:
    - Go, Ge, Lo, Le of 2-site canonical iLPTN   
    """

    # for odd site -> bondTensor = Le - Go - Lo
    @tensor bondTensorO[-1 -2; -3 -4] := Le[-1, 1] * Go[1, -2, -3, 2] * Lo[2, -4]
    @tensor expValO =
        onSiteOp[2, 1] * bondTensorO[3, 4, 2, 5] * conj(bondTensorO[3, 4, 1, 5])
    expValO /= norm(bondTensorO)

    # for even site -> bondTensor = Lo - Ge - Le
    @tensor bondTensorE[-1 -2; -3 -4] := Lo[-1, 1] * Ge[1, -2, -3, 2] * Le[2, -4]
    @tensor expValE =
        onSiteOp[2, 1] * bondTensorE[3, 4, 2, 5] * conj(bondTensorE[3, 4, 1, 5])
    expValE /= norm(bondTensorE)

    if abs(imag(expValO)) < 1e-12 && abs(imag(expValE)) < 1e-12
        return (1 / 2) * (real(expValO) + real(expValE))
    else
        ErrorException("Oops! Complex expectation value is found.")
    end
end

function contractEnvL(X, weightSide, leftT, weightMid, rightT)
    @tensor X[-2; -1] :=
        X[9, 1] *
        weightSide[1, 2] *
        leftT[2, 3, 4, 5] *
        weightMid[5, 6] *
        rightT[6, 7, 8, -1] *
        conj(weightSide[9, 10]) *
        conj(leftT[10, 3, 4, 11]) *
        conj(weightMid[11, 12]) *
        conj(rightT[12, 7, 8, -2])
    return X
end

function contractEnvR(X, weightSide, leftT, weightMid, rightT)
    @tensor X[-1; -2] :=
        X[8, 9] *
        rightT[-1, 1, 2, 3] *
        weightSide[3, 4] *
        leftT[4, 5, 6, 7] *
        weightMid[7, 8] *
        conj(rightT[-2, 1, 2, 12]) *
        conj(weightSide[12, 11]) *
        conj(leftT[11, 5, 6, 10]) *
        conj(weightMid[10, 9])
    return X
end

function leftContraction!(
    transferOpLBond1, weightSide, leftT, weightMid, rightT; niters=300, tol=1e-10
)
    """
    Contract an infinite 2-site unit cell from the left to create transfer operators
    leftT_bond2 and leftT_bond1 using Lanczos method as eigensolver
    """

    # create left transfer operator across bond1
    _, transferOpLBond1 = eigsolve(
        transferOpLBond1, 1, :LM, KrylovKit.Arnoldi(; tol=tol, maxiter=niters)
    ) do x
        contractEnvL(x, weightSide, leftT, weightMid, rightT)
    end
    transferOpLBond1 = transferOpLBond1[1] # extract dominant eigenvector
    transferOpLBond1 = 0.5 * (transferOpLBond1 + transferOpLBond1')
    transferOpLBond1 /= norm(transferOpLBond1)

    # create left transfer operator across bond2
    @tensor transferOpLBond2[-2; -1] :=
        transferOpLBond1[5, 1] *
        weightSide[1, 2] *
        conj(weightSide[5, 6]) *
        leftT[2, 3, 4, -1] *
        conj(leftT[6, 3, 4, -2])
    transferOpLBond2 /= norm(transferOpLBond2)

    return transferOpLBond1, transferOpLBond2
end

function rightContraction!(
    transferOpRBond2, weightSide, leftT, weightMid, rightT; niters=300, tol=1e-10
)
    """
    Contract an infinite 2-site unit cell from the right to create transfer operators
    rightT_OE and rightT_EO
    """

    # create right transfer operator across bond2
    _, transferOpRBond2 = eigsolve(
        transferOpRBond2, 1, :LM, KrylovKit.Arnoldi(; tol=tol, maxiter=niters)
    ) do x
        contractEnvR(x, weightSide, leftT, weightMid, rightT)
    end
    transferOpRBond2 = transferOpRBond2[1] # extract dominant eigenvector
    transferOpRBond2 = 0.5 * (transferOpRBond2 + transferOpRBond2')
    transferOpRBond2 /= norm(transferOpRBond2)

    # create right transfer operator across bond1
    @tensor transferOpRBond1[-1; -2] :=
        transferOpRBond2[4, 6] *
        weightMid[3, 4] *
        leftT[-1, 1, 2, 3] *
        conj(leftT[-2, 1, 2, 5]) *
        conj(weightMid[5, 6])
    transferOpRBond1 /= norm(transferOpRBond1)

    return transferOpRBond2, transferOpRBond1
end

function orthonormalizeiLPTN(
    transferOpL, transferOpR, weightSide, leftT, weightMid, rightT; dtol=1e-12
)
    """
    Return:
    - orthnormalized left and right tensor
    - transferOpL: left normalized by gaugeL
    - transferOpR: right normalized by gaugeR
    """

    # diagonalize left environment tensor
    eval_L, evec_L = eig(0.5 * (transferOpL * transferOpL'))

    # discard small singular values
    evec_L_trial = convert(Array, evec_L)
    eval_L_trial = real(diag(convert(Array, eval_L)))
    ord = sortperm(eval_L_trial; rev=true)
    chiTemp = sum(eval_L_trial .> dtol)
    evec_L_trial = evec_L_trial[:, ord[1:chiTemp]]
    eval_L_trial = eval_L_trial[ord[1:chiTemp]]
    evec_L = TensorMap(evec_L_trial, ComplexSpace(size(evec_L_trial)[1]),ComplexSpace(size(evec_L_trial)[2]))
    eval_L = TensorMap(diagm(eval_L_trial), ComplexSpace(size(eval_L_trial)[1]), ComplexSpace(size(eval_L_trial)[1]))

    @show evec_L * evec_L'
    X = sqrt(eval_L) * evec_L'
    @show X * X'
    @tensor test_trial_1[-2; -1] := X'[1, -1] * transferOpL[2, 1] * X[-2, 2]
    @show test_trial_1

    # diagonalize right environment tensor
    eval_R, evec_R = eig(0.5 * (transferOpR * transferOpR'))
    # discard small singular values
    evec_R_trial = convert(Array, evec_R)
    eval_R_trial = real(diag(convert(Array, eval_R)))
    ord = sortperm(eval_R_trial; rev=true)
    chiTemp = sum(eval_R_trial .> dtol)
    evec_R_trial = evec_R_trial[:, ord[1:chiTemp]]
    eval_R_trial = eval_R_trial[ord[1:chiTemp]]
    evec_R = TensorMap(evec_R_trial, ComplexSpace(size(evec_R_trial)[1]),ComplexSpace(size(evec_R_trial)[2]))
    eval_R = TensorMap(diagm(eval_R_trial), ComplexSpace(size(eval_R_trial)[1]), ComplexSpace(size(eval_R_trial)[1]))
    
    @show evec_R * evec_R'
    Y = evec_R * sqrt(eval_R)
    @show Y' * Y
    @tensor test_trial_2[-1; -2] := Y'[-1, 1] * transferOpR[1, 2] * Y[2, -2]
    @show test_trial_2


    @tensor weightMid[-1; -2] := X[-1, 1] * weightMid[1, 2] * Y[2, -2]

    weightMid /= norm(weightMid)

    U, S, V, _ = tsvd(weightMid, (1,), (2,); alg=TensorKit.SVD())
    weightMid = S / norm(S)

    # build x,y gauge change matrices
    @tensor gaugeL[-1; -2] := evec_L[-1, 1] * pinv(sqrt(eval_L))[1, 2] * U[2, -2]
    @tensor gaugeR[-1; -2] := V[-1, 1] * pinv(sqrt(eval_R))[1, 2] * evec_R'[2, -2]
    # @tensor gaugeL[-1; -2] := evec_L[-1, 1] * pinv(sqrt(eval_L))[1, -2] 
    # @tensor gaugeR[-1; -2] := pinv(sqrt(eval_R))[-1, 2] * evec_R'[2, -2]
    # @tensor gaugeL[-1; -2] := X'[-1, 1] * pinv(sqrt(eval_L))[1, -2]
    # @tensor gaugeR[-1; -2] := pinv(sqrt(eval_R))[-1, 1] * Y'[1, -2]

    @tensor leftT[-1 -2; -3 -4] := leftT[-1, -2, -3, 1] * gaugeL[1, -4]
    @tensor rightT[-1 -2; -3 -4] := gaugeR[-1, 1] * rightT[1, -2, -3, -4]

    @tensor leftT_norm[-1 -2; -3 -4] := weightSide[-1, 1] * leftT[1, -2, -3, 2] * weightMid[2, -4]
    leftT /= norm(leftT_norm)

    @tensor rightT_norm[-1 -2; -3 -4] :=
        weightMid[-1, 1] * rightT[1, -2, -3, 2] * weightSide[2, -4]
    rightT /= norm(rightT_norm)

    return leftT, weightMid, rightT, gaugeL, gaugeR
end