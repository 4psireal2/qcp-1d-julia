using TensorKit
using KrylovKit # Lanczos - real EVal, Oddrnoldi - complex Eval


function contractEnv(X, env)
    """
    Iterative contracting function to find dominant eigenvector := transfer operator
    """
    @tensor X[-1; -2] := env[-1, 2, -2, 1] * X[1, 2];
    return X
end


function findTransferOp(transferOp, env; tol=1e-12, maxiter=1000)
    """
    Contract an infinite 2-site unit cell to create a transfer operator

    Params:
    - transferOp: initial guess of the transfer operator, dim = bondDim x bondDim
    - env: environment to be contracted into
    """
    _, transferOp = eigsolve(transferOp, 1, :LM, KrylovKit.Arnoldi(tol = tol, maxiter = maxiter)) do x contractEnv(x, env) end; 
    transferOp = transferOp[1] / norm(transferOp[1]);  # extract dominant eigenvector

    return transferOp
end


function findGauge(env; isConj::Bool)
    ###XXX: remove negative eigenenvalues?

    eVecs = (env + env') / 2;
    eval, evec = eig(eVecs, (1, ), (2, ));

    evalMat = reshape(convert(Array, eval), (dim(space(eval)[1]), dim(space(eval)[1])));
    if sum(diag(real(evalMat))) < 0
        eval = -eval; # eVecs is equivalent up to overall sign ;
    end
        
    gaugeOp = evec * sqrt(eval);
    if isConj
        gaugeOp = gaugeOp';
    end

    return gaugeOp
end


function orthogonalizeiMPS!(bondTensor, weightMid, weightSide, transferOpL, transferOpR)
    """
    Return left and right tensor over a link such that
    the new transfer operators are orthonormalized
    
    transferOpL_can = (weightSide * tensorL) *  (weightSide * tensorL)' [left-normalised]
    transferOpR_can = (tensorR * weightSide) *  (tensorR * weightSide)' [right-normalised]
    """

    gaugeL = findGauge(transferOpL, isConj=true); # X'
    gaugeR = findGauge(transferOpR, isConj=false); # Y

    @tensor weightSide[-1; -2] := gaugeL[-1, 1] * weightSide[1, 2] * gaugeR[2, -2];
    weightSide /= norm(weightSide);
    
    # orthogonalize coarse-grained bondTensor
    @tensor bondTensor[-1 -2 -3; -4] := gaugeR'[-1, 2] * bondTensor[2, -2, -3, 3] *
                                        gaugeL'[3, -4];

    # decompose bondTensor (Eq. 7 iTEBD.5)
    @tensor bondTensor[-1 -2 -3; -4] := weightSide[-1, 1] * bondTensor[1, -2, -3, 2] * weightSide[2, -4];
    U, S, V, _ = tsvd(bondTensor, (1, 2), (3, 4));
    weightMid = S/norm(S);

    @tensor tensorL[-1 -2; -3] := pinv(weightSide)[-1, 1] * U[1, -2, -3];
    @tensor tensorR[-1 -2; -3] := V[-1, -2, 1] * pinv(weightSide)[1, -3];
    
    return tensorL, tensorR, weightMid, weightSide
end


function applyGate!(leftT, rightT, weightMid, weightSide, op, bondDim, truncErr)
    @tensor bondTensor[-1 -2 -3; -4] := weightSide[-1, 1] * leftT[1, -2, 2] * weightMid[2, 3] * 
                                        rightT[3, -3, 4] * weightSide[4, -4];
    @tensor bondTensor[-1 -2 -3; -4] := op[-2, -3, 1, 2] * bondTensor[-1, 1, 2, -4];
    # bondTensor_mat = reshape(convert(Array, bondTensor), dim(space(bondTensor)[2]), dim(space(bondTensor)[2]));
    # @show bondTensor_mat

    U, S, V, ϵ = tsvd(bondTensor, (1, 2, ), (3, 4, ), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
    
    @tensor leftT[-1 -2; -3] := pinv(weightSide)[-1, 1] * U[1, -2, -3];
    @tensor rightT[-1 -2; -3] := V[-1, -2, 1] * pinv(weightSide)[1, -3];
    weightMid = S / norm(S);

    # updated bondTensor
    @tensor bondTensor[-1 -2 -3; -4] := U[-1, -2, 1] * S[1, 2] * V[2, -3, -4];

    return bondTensor, leftT, rightT, weightMid
end


function computeBondEnergy(bondTensor, leftT, rightT, weightMid, weightSide, op)

    # compute left-bond energy
    @tensor bondL[-1 -2 -3; -4] :=  weightSide[-1, 1] * 
                                    leftT[1, 2, 3] * weightMid[3, 4] * rightT[4, 5, 6] * 
                                    weightSide[6, -4] *
                                    op[-2, -3, 2, 5];
    @tensor bondL[-2; -1] :=    bondL[1, 3, 6, -1] * 
                                conj(weightSide[1, 2]) * 
                                conj(leftT[2, 3, 4]) * conj(weightMid[4, 5]) * conj(rightT[5, 6, 7]) *
                                conj(weightSide[7, -2]);

    @tensor rightSide[-1 -2; -3] := leftT[-1, -2 ,1] * weightMid[1, -3];
    @tensor energyL = bondL[4, 1] * rightSide[1, 2, 3] * conj(rightSide[4, 2, 3]);

    # compute right-bond energy
    @tensor bondR[-1 -2 -3; -4] :=  weightMid[-1, 1] * 
                                    rightT[1, 2, 3] * weightSide[3, 4] * leftT[4, 5, 6] * 
                                    weightMid[6, -4] *
                                    op[-2, -3, 2, 5];

    @tensor bondR[-1; -2] :=    bondR[-1, 1, 2, 3] * 
                                conj(weightMid[-2, 4]) * 
                                conj(rightT[4, 1, 5]) * conj(weightSide[5, 6]) * conj(leftT[6, 2, 7]) *
                                conj(weightMid[7, 3]);
    
    @tensor leftSide[-1 -2; -3] := weightSide[-1, 1] * leftT[1, -2, -3];
    @tensor energyR = bondR[3, 4] * leftSide[1, 2, 3] * conj(leftSide[1, 2, 4]);

    @tensor iNorm = bondTensor[1, 2, 3, 4] * conj(bondTensor[1, 2, 3, 7]) * rightSide[4, 5, 6] * conj(rightSide[7, 5, 6]);


    if abs(imag(energyL)) < 1e-12 && abs(imag(energyR)) < 1e-12 && abs(imag(iNorm)) < 1e-12
        return (1/2) * (energyL + energyR) / iNorm
    else
        ErrorException("Oops! Complex energy or norm is found.")
    end

end


function iTEBD!(Go, Ge, Lo, Le, expHo, expHe, H, bondDim; truncErr=1e-6,)
    """
    2nd order TEBD for unitary evolution for one time step
    """

    # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    bondTensorOdd, Go, Ge, Lo = applyGate!(Go, Ge, Lo, Le, expHo, bondDim, truncErr);
    energyOdd1 = computeBondEnergy(bondTensorOdd, Go, Ge, Lo, Le, H);

    # even bond update -> bondTensor = Lo - Ge - Le - Go - Lo
    bondTensorEven, Ge, Go, Le = applyGate!(Ge, Go, Le, Lo, expHe, bondDim, truncErr);
    energyEven = computeBondEnergy(bondTensorEven, Ge, Go, Le, Lo, H);

    # odd bond update -> bondTensor = Le - Go - Lo - Ge - Le
    # bondTensorOdd, Go, Ge,  Lo = applyGate!(Go, Ge, Lo, Le, expHo, bondDim, truncErr);
    # energyOdd2 = computeBondEnergy(bondTensorOdd, Go, Ge, Lo, Le, H);
    
    # return Go, Ge, Lo, Le, (1/3) * (energyOdd1 + energyEven + energyOdd2)
    return Go, Ge, Lo, Le, (1/2) * (energyOdd1 + energyEven)

        
end