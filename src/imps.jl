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


function orthogonalizeiMPS!(bondTensor, weightSide, transferOpL, transferOpR)
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

    U, S, V, _ = tsvd(weightSide, (1, ), (2, ), alg = TensorKit.SVD());
    
    # orthogonalize coarse-grained bondTensor
    @tensor bondTensor[-1 -2 -3; -4] := V[-1, 1] * gaugeR'[1, 2] * bondTensor[2, -2, -3, 3] *
                                        gaugeL'[3, 4] * U[4, -4];

    # decompose bondTensor (Eq. 7 iTEBD.5)
    @tensor bondTensor[-1 -2 -3; -4] := weightSide[-1, 1] * bondTensor[1, -2, -3, 2] * weightSide[2, -4];
    U, S, V, _ = tsvd(bondTensor, (1, 2), (3, 4), alg = TensorKit.SVD());
    weightMid = S/norm(S);

    @tensor tensorL[-1 -2; -3] := pinv(weightSide)[-1, 1] * U[1, -2, -3];
    @tensor tensorR[-1 -2; -3] := V[-1, -2, 1] * pinv(weightSide)[1, -3];
    
    return tensorL, tensorR, weightMid, weightSide
end