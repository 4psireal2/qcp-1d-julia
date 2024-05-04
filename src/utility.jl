# module Utility

using LinearAlgebra
using MPSKit
using TensorKit


function initializeRandomMPS(N, d::Int64 = 2; bonddim:: Int64 = 1)::Vector{TensorMap}
    """
    Return random vectorized MPO (MPS-like) for N sites. Bond dimension = 1 means 
    product state with minimal entanglement
    """

    randomMPS = Vector{TensorMap}(undef, N);

    # left MPO boundary
    randomMPS[1] = TensorMap(ones, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)',  ComplexSpace(bonddim));
    for i = 2 : (N-1)
        randomMPS[i] = TensorMap(ones, ComplexSpace(bonddim) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)',  ComplexSpace(bonddim));
    end
    randomMPS[N] = TensorMap(ones, ComplexSpace(bonddim) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)', ComplexSpace(1));

    return randomMPS;
end


function initializeBasisMPS(N::Int64, basis::Vector; d::Int64 = 2)::Vector{TensorMap}
    """
    Return vectorized MPO (MPS-like) given basis for N sites. Bond dimension = 1 means 
    product state with minimal entanglement
    """

    mps = Vector{TensorMap}(undef, N);

    for i = 1 : N
        tensor = zeros(ComplexF64, 1, d, d, 1)
        tensor[1, :, :, 1] = basis[i]
        mps[i] = TensorMap(tensor, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d)',ComplexSpace(1));
    end
    
    return mps;
end


function orthogonalizeMPS(mps, orthoCenter::Int)::Vector{TensorMap}

    N = length(mps);
    orthoMPS = deepcopy(mps);

    # bring sites 1 to orthoCenter-1 into left-orthogonal form
    for i = 1 : 1 : (orthoCenter - 1)
        Q, R = leftorth(orthoMPS[i], (1, 2, 3), (4, ), alg = QRpos());

        orthoMPS[i + 0] = Q ;
        orthoMPS[i + 1] = permute(R * permute(orthoMPS[i + 1], (1, ), (2, 3, 4)), (1, 2, 3), (4, ))
    end

    # bring sites orthCenter + 1 to N into right-orthogonal form
    for i = N : -1 : (orthoCenter + 1)
        L, Q = rightorth(orthoMPS[i], (1, ), (2, 3, 4), alg = LQpos());

        orthoMPS[i - 1] = permute(permute(orthoMPS[i-1], (1, 2, 3), (4, )) * L, (1, 2, 3), (4, ));
        orthoMPS[i - 0] = permute(Q, (1, 2, 3), (4, ));
    end

    return orthoMPS;
end


function orthonormalizeMPS(mps::Vector{TensorMap})::Vector{TensorMap}
    orthoMPS = orthogonalizeMPS(mps, 1);
    normMPS = real(tr(orthoMPS[1]' * orthoMPS[1]));
    orthoMPS[1] /= sqrt(normMPS);

    return orthoMPS;
end


function multiplyMPOMPO(mpo1::Vector{TensorMap}, mpo2::Vector{TensorMap})::Vector{TensorMap}
    """
    Compute mpo1*mpo2
    """
    length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    N = length(mpo1);

    fusers = PeriodicArray(map(zip(mpo2, mpo1)) do (mp1, mp2)
        return isometry(fuse(space(mp1, 1), space(mp2, 1)),
                         space(mp1, 1) * space(mp2, 1))
    end)

    resultMPO = Vector{TensorMap}(undef, N);
    for i = 1 : N
        @tensor resultMPO[i][-1 -2 -3; -4 -5 -6] := mpo1[i][1 2 3; -4 -5 4] *
                                           mpo2[i][5 -2 -3; 2 3 6] *
                                           fusers[i][-1; 1 5] *
                                           conj(fusers[i+1][-6; 4 6])
    end

    return resultMPO
end


function computeSiteExpVal_vMPO(mps, mpo)::Vector
    """ 
    Compute expectation value <a> = (1/N) . ∑_k Tr(ρA_k)
    """

    # get length of mps
    N = length(mps);

    # compute expectation values
    expVals = zeros(Float64, N);
    for i = 1 : N
        boundaryL = TensorMap(ones, one(ComplexSpace()), ℂ^1);
        boundaryR = TensorMap(ones, ℂ^1, one(ComplexSpace()));
        for j = 1 :N
            if j==i
                _boundaryL = TensorMap(ones, one(ComplexSpace()), space(mpo)[1]);
                _boundaryR = TensorMap(ones, conj(space(mpo)[6]), one(ComplexSpace()));
                @tensor _mpo[-3 -4; -1 -2] := _boundaryL[1] * mpo[1,-3,-4,-1,-2,2] * _boundaryR[2];
                @tensor contraction[-1 -2;-3 -4] := mps[j][-1, 1, 2, -4] * _mpo[-2,-3, 1, 2];
                dim1, dim4 = dim(space(contraction)[1]), dim(space(contraction)[4]);
                _contraction = reshape(convert(Array, contraction), (dim1,2,2,dim4));
                contraction = TensorMap(sum(_contraction, dims=[2,3]), space(contraction)[1], conj(space(contraction)[4]));
            else
                dim1, dim4 = dim(space(mps[j])[1]), dim(space(mps[j])[4]);
                _contraction = reshape(convert(Array, mps[j]), (dim1,2,2,dim4));
                contraction = TensorMap(sum(_contraction, dims=[2,3]), space(mps[j])[1], conj(space(mps[j])[4]));
            end
            boundaryL = boundaryL * contraction # @tensor boundaryL[-1] := boundaryL[1] * contraction[1, -1]
        end

        expVals[i] = tr(boundaryL * boundaryR)
    end
    
    return expVals;
end


function computeExpVal(mps, mpo)::Float64
    """ 
    Compute expectation value Tr(ρ† . A . ρ)
    """
    N = length(mps);
    mps = orthonormalizeMPS(mps);

    # contract from left to right
    boundaryL = TensorMap(ones, space(mps[1], 1), space(mps[1], 1) ⊗ space(mpo[1], 1));
    for i = 1 : 1 : N
        @tensor boundaryL[-1; -2 -3] := boundaryL[1, 7, 4] * mps[i][7, 5, 6, -2] * mpo[i][4, 2, 3, 5, 6, -3] * conj(mps[i][1, 2, 3, -1]);
    end

    boundaryR = TensorMap(ones, space(mps[N], 4)' ⊗ space(mpo[N], 6)', space(mps[N], 4)');            

    # contract to get expectation value
    expVal = tr(boundaryL * boundaryR);
    if abs(imag(expVal)) < 1e-12
        expVal = real(expVal);
    else
        ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
    end
    
    return expVal;
end


function computeSiteExpVal_mps(mps, onsiteOp)
    """ 
    Compute the expectation value < Ψ | onsiteOp | Ψ > for each site of the MPS 
    """

    # get length of mps
    N = length(mps);

    # compute Hermitian part of MPO
    hermitMPS = Vector{TensorMap}(undef, N);
    for i = 1 : N
        mps_dag_i = TensorMap(conj(convert(Array, mps[i])), codomain(mps[i]), domain(mps[i]));
        hermitMPS[i] = mps_dag_i + mps[i] 
    end
    hermitMPS /= 2

    # compute expectation values
    _boundaryL = TensorMap(ones, one(ComplexSpace()), space(onsiteOp)[1]);
    _boundaryR = TensorMap(ones, conj(space(onsiteOp)[6]), one(ComplexSpace()));
    @tensor onsiteOp[-3 -4; -1 -2] := _boundaryL[1] * onsiteOp[1,-3,-4,-1,-2,2] * _boundaryR[2]

    expVals = zeros(Float64, N);
    for i = 1 : N

        # bring MPS into canonical form
        hermitMPS = orthogonalizeMPS(hermitMPS, i); 
        psiNormSq = real(tr(hermitMPS[i]' * hermitMPS[i]));
        
        # compute expectation value
        
        expVal = @tensor conj(hermitMPS[i][-1, 2, 3, -6]) * onsiteOp[2, 3, 4, 5] * hermitMPS[i][-1, 4, 5, -6];
        if abs(imag(expVal)) < 1e-12
            expVal = real(expVal);
            expVals[i] = expVal / psiNormSq;
        else
            ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
        end
    end

    return expVals;
end


function addMPSMPS(mps1::Vector{TensorMap}, mps2::Vector{TensorMap})::Vector{TensorMap}
    """
    Add 2 MPS of the same bond dimension
    """
    resultMPS = Vector{TensorMap}(undef, N);
    
    for i = 1 : N
        dim1, dim2, dim3, dim4 = dim(space(mps1[i])[1]), 
                                 dim(space(mps1[i])[2]) + dim(space(mps2[i])[2]),
                                 dim(space(mps1[i])[3]) + dim(space(mps2[i])[3]),
                                 dim(space(mps1[i])[4]);
        tensor = zeros(ComplexF64,dim1, dim2, dim3, dim4)
        # tensor[1, :, :, 1] = 
        resultMPS[i] = TensorMap(tensor, ComplexSpace(dim1) ⊗ ComplexSpace(dim2) ⊗ ComplexSpace(dim3)', ComplexSpace(dim4));


    end
    return resultMPS        
end