"""
Ref: Positive Tensor Network Approach for Simulating Open Quantum Many-Body Systems
https://doi.org/10.1103/PhysRevLett.116.237201
"""

using LinearAlgebra
using TensorKit
using MPSKit


function createXOnes(N::Int64; d::Int64 = 2, bondDim::Int64 = 1,  krausDim::Int64 = 1)

    X = Vector{TensorMap}(undef, N);

    X[1] = TensorMap(ones, ComplexSpace(1) ⊗ ComplexSpace(krausDim),  ComplexSpace(d) ⊗ ComplexSpace(bondDim));
    for i = 2 : (N-1)
        X[i] = TensorMap(ones, ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim),  ComplexSpace(d) ⊗ ComplexSpace(bondDim));
    end 
    X[N] = TensorMap(ones, ComplexSpace(bondDim) ⊗ ComplexSpace(krausDim),  ComplexSpace(d) ⊗ ComplexSpace(1));

    return X
end


function createXBasis(N::Int64, basis; d::Int64 = 2, bondDim::Int64 = 1,  krausDim::Int64 = 1)
    X = Vector{TensorMap}(undef, N);
    for i = 1 : N
        tensorBase = zeros(ComplexF64, 1, krausDim, d, 1);
        tensorBase[:, :, 1, 1] = reshape([basis[i][1]], 1, 1);
        tensorBase[:, :, 2, 1] = reshape([basis[i][2]], 1, 1);
        X[i] = TensorMap(tensorBase, ComplexSpace(1) ⊗ ComplexSpace(krausDim), ComplexSpace(d) ⊗ ComplexSpace(1));

    end

    return X
end


function multiplyMPOMPO(mpo1::Vector{TensorMap}, mpo2::Vector{TensorMap})
    """
    Compute mpo1*mpo2
    """
    length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    N = length(mpo1);

    fusers = PeriodicArray(map(zip(mpo1, mpo2)) do (mp1, mp2)
        return isometry(fuse(space(mp1, 1), space(mp2, 1)),
                         space(mp1, 1) * space(mp2, 1))
    end)

    resultMPO = Vector{TensorMap}(undef, N);
    for i = 1 : N
        @tensor resultMPO[i][-1 -2; -3 -4] := mpo1[i][1 2; -3 4] *
                                           mpo2[i][3 -2; 2 5] *
                                           fusers[i][-1; 1 3] *
                                           conj(fusers[i+1][-4; 4 5]);
    end

    return resultMPO
end


function orthogonalizeX(X, orthoCenter::Int = 1)::Vector{TensorMap}
    """
    Params
    X: MPO
    """
    N = length(X);
    orthoX = deepcopy(X);

    # bring sites 1 to orthoCenter-1 into left-orthogonal form
    
    for i = 1 : 1 : (orthoCenter - 1)
        Q, R = leftorth(orthoX[i], (1, 2, 3), (4, ), alg = QRpos());

        orthoX[i + 0] = permute(Q, (1, 2), (3, 4)) ;
        orthoX[i + 1] = permute(R * permute(orthoX[i + 1], (1, ), (2, 3, 4)), (1, 2), (3, 4))
    end

    # bring sites orthCenter + 1 to N into right-orthogonal form
    for i = N : -1 : (orthoCenter + 1)
        L, Q = rightorth(orthoX[i], (1, ), (2, 3, 4), alg = LQpos());

        orthoX[i - 1] = permute(permute(orthoX[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
        orthoX[i - 0] = permute(Q, (1, 2), (3, 4));
    end

    return orthoX
end


function orthonormalizeX(X)
    orthoX = orthogonalizeX(X, 1);
    normX = real(tr(orthoX[1]' * orthoX[1]));
    orthoX[1] /= sqrt(normX);

    return orthoX;
end


function computeNorm(X)::Float64
    N = length(X);

    boundaryL = TensorMap(ones, ℂ^1, ℂ^1);
    boundaryR = TensorMap(ones, ℂ^1, ℂ^1);

    for i = 1 : N
        @tensor boundaryL[-1; -2] := boundaryL[1, 2] * conj(X[i][1, 3, 4, -1]) * X[i][2, 3, 4, -2];
    end

    lptnNorm = tr(boundaryL * boundaryR)
    
    if abs(imag(lptnNorm)) < 1e-12
        return real(lptnNorm)
    else
        ErrorException("Complex norm is found.")
    end
end


function computePurity(X)::Float64
    # XXX: a more efficient way ?
    N = length(X);

    boundaryL = TensorMap(ones, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1);
    boundaryR = TensorMap(ones, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1);

    for i = 1 : N
        @tensor boundaryL[-1 -2; -3 -4] := boundaryL[4, 8, 1, 6] * X[i][1, 2, 3, -3] * conj(X[i][4, 2, 5, -1]) * X[i][6, 7, 5, -4] * conj(X[i][8, 7, 3, -2]);
    end

    purity = tr(boundaryL * boundaryR)
    
    if abs(imag(purity)) < 1e-12
        return purity
    else
        ErrorException("Complex purity is found.")
    end
end


function computeSiteExpVal(X, onSiteOp)
    N = length(X);

    lptnNorm = computeNorm(X);
    expVals = zeros(Float64, N);

    for i = 1 : N

        boundaryL = TensorMap(ones, ℂ^1, ℂ^1);
        boundaryR = TensorMap(ones, ℂ^1, ℂ^1);

        for j in 1 : N
            if j==i
                @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 5, -1]) * X[j][4, 1, 2, -2] * onSiteOp[2, 5];
            else
                @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 2, -1]) * X[j][4, 1, 2, -2];
            end
        end

        expVal = tr(boundaryL * boundaryR);
        if abs(imag(expVal)) < 1e-12
            expVals[i] = real(expVal) / lptnNorm;
        else
            ErrorException("Complex expectation value is found.")
        end
    end
    
    return expVals, sum(expVals)/N
end


function densDensCorr(r::Int64, X, onSiteOp)
    """
    Compute  <O_{r} . O_{0}> - <O_{0}>^2 with trace
    """

    N = length(X);
    lptnNorm = computeNorm(X);

    # compute <O_{r} . O_{0}>
    boundaryL = TensorMap(ones, ℂ^1, ℂ^1);
    boundaryR = TensorMap(ones, ℂ^1, ℂ^1);

    for j in 1 : N
        if j==1 ||  j==r
            @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 5, -1]) * X[j][4, 1, 2, -2] * onSiteOp[2, 5];
        else
            @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 2, -1]) * X[j][4, 1, 2, -2];
        end
    end

    meanProduct = tr(boundaryL * boundaryR);
    if abs(imag(meanProduct)) < 1e-12
        meanProduct = real(meanProduct) / lptnNorm;
    else
        ErrorException("Complex expectation value is found.")
    end

    # compute <O_{0}>^2
    boundaryL = TensorMap(ones, ℂ^1, ℂ^1);
    boundaryR = TensorMap(ones, ℂ^1, ℂ^1);

    for j in 1 : N
        if j==0
            @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 5, -1]) * X[j][4, 1, 2, -2] * onSiteOp[2, 5];
        else
            @tensor boundaryL[-1; -2] := boundaryL[3, 4] * conj(X[j][3, 1, 2, -1]) * X[j][4, 1, 2, -2];
        end
    end
    productMean = tr(boundaryL * boundaryR);
    if abs(imag(productMean)) < 1e-12
        productMean = (real(productMean) / lptnNorm)^2;
    else
        ErrorException("Complex expectation value is found.")
    end

    return meanProduct - productMean
end


function computeEntSpec(X)
    """
    Compute entanglement spectrum for the bipartion of LPTN chain
    """

    N = length(X);
    indL, indR = N÷2, N÷2 + 1;
    @tensor bondTensor[-1 -4 -7 -8; -2 -3 -5 -6] := X[indL][-1, 1, -2, 2] * conj(X[indL][-3, 1, -4, 3]) * X[indR][2, 4, -5, -6] * conj(X[indR][3, 4, -7, -8]);
    
    U, S, V, ϵ = tsvd(bondTensor, (1, 2, 5, 6), (3, 4, 7, 8), alg = TensorKit.SVD());
    
    return S
end