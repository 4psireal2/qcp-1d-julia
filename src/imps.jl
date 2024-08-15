using LinearAlgebra
using TensorKit
using KrylovKit # Lanczos - real EVal, Oddrnoldi - complex Eval

function contractEnvL(X, weightSide, leftT, weightMid, rightT)
    @tensor X[-2; -1] :=
        X[7, 1] *
        weightSide[1, 2] *
        leftT[2, 3, 4] *
        weightMid[4, 5] *
        rightT[5, 6, -1] *
        conj(weightSide[7, 8]) *
        conj(leftT[8, 3, 9]) *
        conj(weightMid[9, 10]) *
        conj(rightT[10, 6, -2])
    return X
end

function contractEnvR(X, weightSide, leftT, weightMid, rightT)
    @tensor X[-1; -2] :=
        X[6, 10] *
        rightT[-1, 1, 2] *
        weightSide[2, 3] *
        leftT[3, 4, 5] *
        weightMid[5, 6] *
        conj(rightT[-2, 1, 7]) *
        conj(weightSide[7, 8]) *
        conj(leftT[8, 4, 9]) *
        conj(weightMid[9, 10])
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
        transferOpLBond1[4, 1] *
        weightSide[1, 2] *
        conj(weightSide[4, 5]) *
        leftT[2, 3, -1] *
        conj(leftT[5, 3, -2])
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
        transferOpRBond2[3, 5] *
        weightMid[2, 3] *
        leftT[-1, 1, 2] *
        conj(leftT[-2, 1, 4]) *
        conj(weightMid[4, 5])
    transferOpRBond1 /= norm(transferOpRBond1)

    return transferOpRBond2, transferOpRBond1
end

function orthonormalizeiMPS(
    transferOpL, transferOpR, weightSide, leftT, weightMid, rightT; dtol=1e-12
)
    """
    Return:
    - orthnormalized left and right tensor
    - transferOpL: left normalized by gaugeL
    - transferOpR: right normalized by gaugeR
    """

    # diagonalize left environment tensor
    eval_L_trial, eval_L_trial = eig(0.5 * (transferOpL * transferOpL'))
    envL_mat = reshape(
        convert(Array, transferOpL), dim(space(transferOpL, 1)), dim(space(transferOpL, 1))
    )
    dtemp = eigen(0.5 * (envL_mat + envL_mat'))
    ord = sortperm(real(dtemp.values); rev=true)
    chitemp = sum(real(dtemp.values) .> dtol)
    UL = dtemp.vectors[:, ord[1:chitemp]]
    DL = dtemp.values[ord[1:chitemp]]

    # envL = evec_L* eval_L * evec_L'
    evec_L = TensorMap(UL, ComplexSpace(size(UL)[1]), ComplexSpace(size(UL)[2]))
    eval_L = TensorMap(diagm(DL), ComplexSpace(size(DL)[1]), ComplexSpace(size(DL)[1]))
    X = sqrt(eval_L) * evec_L'

    # diagonalize right environment tensor
    envR_mat = reshape(
        convert(Array, transferOpR), dim(space(transferOpR, 1)), dim(space(transferOpR, 1))
    )
    dtemp = eigen(0.5 * (envR_mat + envR_mat'))
    ord = sortperm(real(dtemp.values); rev=true)
    chitemp = sum(real(dtemp.values) .> dtol)
    UR = dtemp.vectors[:, ord[1:chitemp]]
    DR = dtemp.values[ord[1:chitemp]]

    # envR = evec_R * eval_R * evec_R'
    evec_R = TensorMap(UR, ComplexSpace(size(UR)[1]), ComplexSpace(size(UR)[2]))
    eval_R = TensorMap(diagm(DR), ComplexSpace(size(DR)[1]), ComplexSpace(size(DR)[1]))
    Y = evec_R * sqrt(eval_R)

    @tensor weightMid[-1; -2] := X[-1, 1] * weightMid[1, 2] * Y[2, -2]

    weightMid /= norm(weightMid)

    U, S, V, _ = tsvd(weightMid, (1,), (2,); alg=TensorKit.SVD())
    weightMid = S / norm(S)

    # build x,y gauge change matrices
    @tensor gaugeL[-1; -2] := evec_L[-1, 1] * pinv(sqrt(eval_L))[1, 2] * U[2, -2]
    @tensor gaugeR[-1; -2] := V[-1, 1] * pinv(sqrt(eval_R))[1, 2] * evec_R'[2, -2]

    @tensor leftT[-1 -2; -3] := leftT[-1, -2, 1] * gaugeL[1, -3]
    @tensor rightT[-1 -2; -3] := gaugeR[-1, 1] * rightT[1, -2, -3]

    @tensor leftT_norm[-1 -2; -3] := weightSide[-1, 1] * leftT[1, -2, 2] * weightMid[2, -3]
    leftT /= norm(leftT_norm)

    @tensor rightT_norm[-1 -2; -3] :=
        weightMid[-1, 1] * rightT[1, -2, 2] * weightSide[2, -3]
    rightT /= norm(rightT_norm)

    return leftT, weightMid, rightT, gaugeL, gaugeR
end

function compute1SiteExpVal(Go, Ge, Lo, Le, onSiteOp)
    """
    Args:
    - Go, Ge, Lo, Le of 2-site canonical iMPS    
    """

    # for odd site -> bondTensor = Le - Go - Lo
    @tensor bondTensorO[-1 -2; -3] := Le[-1, 1] * Go[1, -2, 2] * Lo[2, -3]
    @tensor expValO = onSiteOp[4, 2] * bondTensorO[1, 2, 3] * conj(bondTensorO[1, 4, 3])
    expValO /= norm(bondTensorO)

    # for even site -> bondTensor = Lo - Ge - Le
    @tensor bondTensorE[-1 -2; -3] := Lo[-1, 1] * Ge[1, -2, 2] * Le[2, -3]
    @tensor expValE = onSiteOp[4, 2] * bondTensorE[1, 2, 3] * conj(bondTensorE[1, 4, 3])
    expValE /= norm(bondTensorE)

    if abs(imag(expValO)) < 1e-12 && abs(imag(expValE)) < 1e-12
        return (1 / 2) * (real(expValO) + real(expValE))
    else
        ErrorException("Oops! Complex expectation value is found.")
    end
end

function compute2SiteExpVal(Go, Ge, Lo, Le, oP2)
    """
    Args:
    - Go, Ge, Lo, Le of 2-site canonical iMPS        
    """

    # expVal for odd bond
    @tensor bondTensorO[-1 -2 -3; -4] :=
        Le[-1, 1] * Go[1, -2, 2] * Lo[2, 3] * Ge[3, -3, 4] * Le[4, -4]
    @tensor expValO =
        bondTensorO[1, 2, 3, 4] * oP2[5, 6, 2, 3] * conj(bondTensorO[1, 5, 6, 4])
    expValO /= norm(bondTensorO)

    # expVal for even bond
    @tensor bondTensorE[-1 -2 -3; -4] :=
        Lo[-1, 1] * Ge[1, -2, 2] * Le[2, 3] * Go[3, -3, 4] * Lo[4, -4]
    @tensor expValE =
        bondTensorE[1, 2, 3, 4] * oP2[5, 6, 2, 3] * conj(bondTensorE[1, 5, 6, 4])
    expValE /= norm(bondTensorE)

    if abs(imag(expValO)) < 1e-12 && abs(imag(expValE)) < 1e-12
        return (1 / 2) * (real(expValO) + real(expValE))
    else
        @show expValO
        @show expValE
        ErrorException("Oops! Complex expectation value is found.")
    end
end

function computeCorrLen(Go, Ge, Lo, Le)
    bondDim = dim(space(Lo, 1))

    if bondDim > 100
        println("Bond dimension is too large...")
    end

    @tensor LoGo[-1 -2; -3] := Go[-1, -2, 1] * Lo[1, -3]
    @tensor LoGo[-1 -4; -2 -3] := LoGo[-1, 1, -2] * conj(LoGo[-3, 1, -4])

    @tensor LeGe[-1 -2; -3] := Ge[-1, -2, 1] * Le[1, -3]
    @tensor LeGe[-1 -4; -2 -3] := LeGe[-1, 1, -2] * conj(LeGe[-3, 1, -4])

    @tensor transferOp[-1 -4; -2 -3] := LoGo[-1, 2, 1, -3] * LeGe[1, -4, -2, 2]

    dimMat = dim(space(LoGo, 1)) * dim(space(LeGe, 1))
    transferMat = real(reshape(convert(Array, transferOp), dimMat, dimMat))

    eigvals, _ = eigsolve(transferMat, dimMat, 2, :LM)
    corrLength = -2 / log(abs(eigvals[2] / eigvals[1]))

    return corrLength
end
