import LinearAlgebra: kron, norm, dot, tr

import Printf: @printf, @sprintf
import KrylovKit: eigsolve, Lanczos
import MPSKit: PeriodicArray
import TensorKit: @tensor, ⊗, fuse, isometry, leftorth, permute, randn, rightorth, space, 
                  truncerr, truncdim, tsvd, ComplexSpace, SVD, TensorMap, LQpos, RQpos


function initializeRandomMPS(N, d::Int64 = 2; bonddim:: Int64 = 1)::Vector{TensorMap}
    """
    Return random vectorized MPO (MPS-like) for N sites. Bond dimension = 1 means 
    product state with minimal entanglement
    """

    randomMPS = Vector{TensorMap}(undef, N);

    # left MPO boundary
    randomMPS[1] = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d),  ComplexSpace(bonddim));
    for i = 2 : (N-1)
        randomMPS[i] = TensorMap(randn, ComplexSpace(bonddim) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d),  ComplexSpace(bonddim));
    end
    randomMPS[N] = TensorMap(randn, ComplexSpace(bonddim) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d), ComplexSpace(1));

    return randomMPS;
end


function initializeBasisMPS(N::Int64, basis::Vector; d::Int64 = 2)::Vector{TensorMap}
    """
    Return vectorized MPO (MPS-like) given basis for N sites. Bond dimension = 1 means 
    product state with minimal entanglement
    """

    mps = Vector{TensorMap}(undef, N);

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, d, d, 1)
    leftB[:, :, 1, 1] = basis[1]
    leftB[:, :, 2, 1] = basis[1]
    mps[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d),  ComplexSpace(1));
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 1, 2, 2, 1)
        bulk[:, :, 1, 1] = basis[i]
        bulk[:, :, 2, 1] = basis[i]
        mps[i] = TensorMap(bulk, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d),  ComplexSpace(1));
    end
    rightB = zeros(ComplexF64, 1, 2, 2, 1)
    rightB[:, :, 1, 1] = basis[N]
    rightB[:, :, 2, 1] = basis[N]
    mps[N] = TensorMap(rightB, ComplexSpace(1) ⊗ ComplexSpace(d) ⊗ ComplexSpace(d), ComplexSpace(1));

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

    # bring sites orthCenter + 1 to N into right-canonical form
    for i = N : -1 : (orthoCenter + 1)
        L, Q = rightorth(orthoMPS[i], (1, ), (2, 3, 4), alg = LQpos());

        orthoMPS[i - 1] = permute(permute(orthoMPS[i-1], (1, 2, 3), (4, )) * L, (1, 2, 3), (4, ));
        orthoMPS[i - 0] = Q;
    end

    return orthoMPS;
end


function orthonormalizeMPS(mps::Vector{TensorMap})::Vector{TensorMap}
    orthoMPS = orthogonalizeMPS(mps, 1);
    normMPS = real(tr(orthoMPS[1]' * orthoMPS[1]));
    orthoMPS[1] /= sqrt(normMPS);

    return orthoMPS;
end


function constructLindbladMPO(omega::Float64, gamma::Float64, N::Int64)::Vector{TensorMap}
    # define operators
    Id = [+1 0 ; 0 +1];

    sigmaX = 0.5 * [0 +1 ; +1 0];
    sigmaXR = kron(sigmaX, Id);
    sigmaXL = kron(Id, sigmaX);

    numberOp = [0 0; 0 1];
    numberOpR = kron(numberOp, Id);
    numberOpL = kron(Id, numberOp);

    annihilationOp = [0 1; 0 0];

    onSite = gamma * kron(annihilationOp, annihilationOp) - (1/2)*numberOpR - (1/2)*numberOpL

    lindbladMPO = Vector{TensorMap}(undef, N);

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 6)
    leftB[1, :, :, 1] = kron(Id, Id);
    leftB[1, :, :, 2] = sigmaXR;
    leftB[1, :, :, 3] = sigmaXL;
    leftB[1, :, :, 4] = numberOpR;
    leftB[1, :, :, 5] = numberOpL;
    leftB[1, :, :, 6] = gamma*onSite;
    lindbladMPO[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(6));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 6, 4, 4, 6);
        bulk[1, :, :, 1] = kron(Id, Id);
        bulk[1, :, :, 2] = sigmaXR;
        bulk[1, :, :, 3] = sigmaXL;
        bulk[1, :, :, 4] = numberOpR;
        bulk[1, :, :, 5] = numberOpL;
        bulk[1, :, :, 6] = gamma * onSite;
        bulk[2, :, :, 6] = -1im * omega * numberOpR;
        bulk[3, :, :, 6] = -1im * omega * numberOpL;
        bulk[4, :, :, 6] = -1im * omega * sigmaXR;
        bulk[5, :, :, 6] = -1im * omega * sigmaXL;
        bulk[6, :, :, 6] = kron(Id, Id);
        
        lindbladMPO[i] = TensorMap(bulk, ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(6));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 6, 4, 4, 1)
    rightB[1, :, :, 1] = gamma*onSite;
    rightB[2, :, :, 1] = -1im * omega * numberOpR;
    rightB[3, :, :, 1] = -1im * omega * numberOpL;
    rightB[4, :, :, 1] = -1im * omega * sigmaXR;
    rightB[5, :, :, 1] = -1im * omega * sigmaXL;
    rightB[6, :, :, 1] = kron(Id, Id);
    lindbladMPO[N] = TensorMap(rightB, ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    return lindbladMPO;
end


function constructLindbladDagMPO(omega::Float64, gamma::Float64, N::Int64)::Vector{TensorMap}
    """
    Construct Lindbladian superoperator
    """
    # define operators
    Id = [+1 0 ; 0 +1];

    sigmaX = 0.5 * [0 +1 ; +1 0];
    sigmaXR = kron(sigmaX, Id);
    sigmaXL = kron(Id, sigmaX);

    numberOp = [0 0; 0 1];
    numberOpR = kron(numberOp, Id);
    numberOpL = kron(Id, numberOp);

    creationOp = [0 0; 1 0];

    onSiteDagger = gamma * kron(creationOp, creationOp) - (1/2)*numberOpR - (1/2)*numberOpL

    lindbladDagMPO = Vector{TensorMap}(undef, N);

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 6)
    leftB[1, :, :, 1] = kron(Id, Id);
    leftB[1, :, :, 2] = sigmaXR;
    leftB[1, :, :, 3] = sigmaXL;
    leftB[1, :, :, 4] = numberOpR;
    leftB[1, :, :, 5] = numberOpL;
    leftB[1, :, :, 6] = gamma*onSiteDagger;
    lindbladDagMPO[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(6));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 6, 4, 4, 6);
        bulk[1, :, :, 1] = kron(Id, Id);
        bulk[1, :, :, 2] = sigmaXR;
        bulk[1, :, :, 3] = sigmaXL;
        bulk[1, :, :, 4] = numberOpR;
        bulk[1, :, :, 5] = numberOpL;
        bulk[1, :, :, 6] = gamma * onSiteDagger;
        bulk[2, :, :, 6] = im * omega * numberOpR;
        bulk[3, :, :, 6] = im * omega * numberOpL;
        bulk[4, :, :, 6] = im * omega * sigmaXR;
        bulk[5, :, :, 6] = im * omega * sigmaXL;
        bulk[6, :, :, 6] = kron(Id, Id);
        
        lindbladDagMPO[i] = TensorMap(bulk, ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(6));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 6, 4, 4, 1)
    rightB[1, :, :, 1] = gamma*onSiteDagger;
    rightB[2, :, :, 1] = im * omega * numberOpR;
    rightB[3, :, :, 1] = im * omega * numberOpL;
    rightB[4, :, :, 1] = im * omega * sigmaXR;
    rightB[5, :, :, 1] = im * omega * sigmaXL;
    rightB[6, :, :, 1] = kron(Id, Id);
    lindbladDagMPO[N] = TensorMap(rightB, ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    return lindbladDagMPO;
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


function updateMPOEnvL(mpoEnvL::TensorMap, mpsT::TensorMap, mpo::TensorMap, mpsB::TensorMap)::TensorMap
    @tensor newEnvL[-1; -2 -3] := mpoEnvL[1, 7, 4] * mpsT[7, 5, 6, -2] * mpo[4, 2, 3, 5, 6, -3] * conj(mpsB[1, 2, 3, -1]);
    
    return newEnvL;
end


function updateMPOEnvR(mpoEnvR::TensorMap, mpsT::TensorMap, mpo::TensorMap, mpsB::TensorMap)::TensorMap
    @tensor newEnvR[-1 -2; -3] := mpsT[-1, 2, 3, 1] * mpo[-2, 5, 6, 2, 3, 4] * conj(mpsB[-3, 5, 6, 7]) * mpoEnvR[1, 4, 7];
    
    return newEnvR;
end


function applyH2(X::TensorMap, EnvL::TensorMap, mpo1::TensorMap, mpo2::TensorMap, EnvR::TensorMap)::TensorMap
    @tensor X[-1 -2 -3; -4 -5 -6] := EnvL[-1, 2, 1] * X[2, 3, 4, 6, 7, 8] * mpo1[1, -2, -3, 3, 4, 5] * mpo2[5, -4, -5, 6, 7, 9] * EnvR[8, 9, -6];
    
    return X;
end


function initializeMPOEnvs(mps::Vector{TensorMap}, mpo::Vector{TensorMap}; centerPos::Int64=1)
    """
    Create left and right environments for DMRG2
    """
    length(mps) == length(mpo) || throw(ArgumentError("dimension mismatch"))

    N = length(mps);

    mpoEnvL = Vector{TensorMap}(undef, N);
    mpoEnvR = Vector{TensorMap}(undef, N);

    mpoEnvL[1] = TensorMap(ones, space(mps[1], 1), space(mps[1], 1) ⊗ space(mpo[1], 1));
    mpoEnvR[N] = TensorMap(ones, space(mps[N], 4)' ⊗ space(mpo[N], 6)', space(mps[N], 4)');

    # compute mpoEnvL up to (centerPos - 1)
    for i = 1 : 1 : (centerPos - 1)
        mpoEnvL[i + 1] = updateMPOEnvL(mpoEnvL[i], mps[i], mpo[i], mps[i]);
    end

    # compute mpoEnvR up to (centerPos + 1)
    for i = N : -1 : (centerPos + 1)
        mpoEnvR[i - 1] = updateMPOEnvR(mpoEnvR[i], mps[i], mpo[i], mps[i]);
    end

    return mpoEnvL, mpoEnvR;
end


function DMRG2(mps::Vector{TensorMap}, mpo::Vector{TensorMap};
               bondDim::Int64 = 20, truncErr::Float64 = 1e-6, convTolE::Float64 = 1e-6,
               eigsTol::Float64 = 1e-16, maxIterations::Int64 = 1, verbosePrint::Bool = false)
    
    N = length(mps);

    initialEnergy = computeExpVal(mps, mpo);
    mpsEnergy = Float64[initialEnergy];
    
    mpoEnvL, mpoEnvR = initializeMPOEnvs(mps, mpo);

    # main DMRG loop
    loopCounter = 1;
    runOptimizationDMRG = true;
    while runOptimizationDMRG
        
        # sweep L ---> R
        for i = 1 : 1 : (N - 1)

            # construct intial bond tensor
            bondTensor = mps[i] * permute(mps[i+1], (1, ), (2, 3, 4));

            # optimize wave function to get newAC
            _, eigenVec = eigsolve(bondTensor, 1, :SR, Lanczos(tol = eigsTol, maxiter = maxIterations)) do x # 1 eigenVal
                applyH2(x, mpoEnvL[i], mpo[i], mpo[i + 1], mpoEnvR[i + 1])
            end

            newBondTensor = eigenVec[1];

            #  perform SVD and truncate to desired bond dimension
            U, S, V, ϵ = tsvd(newBondTensor, (1, 2, 3), (4, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = SVD());
            U = permute(U, (1, 2, 3), (4, ));
            V = permute(S * V, (1, 2, 3), (4, ));

            mps[i], mps[i+1] = U, V;

            # update mpoEnvL
            mpoEnvL[i + 1] = updateMPOEnvL(mpoEnvL[i], mps[i], mpo[i], mps[i]);
        end

        # sweep R --> L
        for i = (N-1) : -1 : 1
            # construct intial bond tensor
            bondTensor = mps[i] * permute(mps[i+1], (1, ), (2, 3, 4));

            # optimize wave function to get newAC
            _, eigenVec = eigsolve(bondTensor, 1, :SR, Lanczos(tol = eigsTol, maxiter = maxIterations)) do x # 1 eigenVal
                applyH2(x, mpoEnvL[i], mpo[i], mpo[i + 1], mpoEnvR[i + 1])
            end

            newBondTensor = eigenVec[1];

            #  perform SVD and truncate to desired bond dimension
            U, S, V, ϵ = tsvd(newBondTensor, (1, 2, 3), (4, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = SVD());
            U = permute(U * S, (1, 2, 3), (4, ));
            V = permute(V, (1, 2, 3), (4, ));

            # update mpoEnvR
            mpoEnvR[i] = updateMPOEnvR(mpoEnvR[i+1], mps[i+1], mpo[i+1], mps[i+1]);
        end

        # normalize 
        mpsNorm = tr(mps[1]' * mps[1]);
        mps[1] /= sqrt(mpsNorm)

        # compute expectation value
        expVal = computeExpVal(mps, mpo);
        if abs(imag(expVal)) < 1e-12
            expVal = real(expVal);
        else
            ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
        end
        mpsEnergy = vcat(mpsEnergy, expVal)

        # check convergence of ground state energy
        energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
        verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, expVal, energyConvergence);
        if energyConvergence < convTolE
            runOptimizationDMRG = false;
        end

        loopCounter += 1;
    end

    # return optimized mps
    finalEnergy = mpsEnergy[end];
    return mps, finalEnergy;
end


function computeExpVal(mps::Vector{TensorMap}, mpo::Vector{TensorMap})::Float64
    """
    Compute expectation value for the whole chain
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


function computeSiteExpVal(mps::Vector{TensorMap}, onsiteOps::Vector{TensorMap})::Vector
    """ 
    Compute the expectation value < psi | onsiteOp | psi > for each site of the MPS 
    """

    # get length of mps
    N = length(mps);

    # compute Hermitian part of MPO
    hermitMPS = Vector{TensorMap}(undef, N);
    for i = 1 : N
        mps_dag_i = TensorMap(conj(convert(Array, mps[i])), codomain(mps[i]), domain(mps[i])); #XXX: Is this correct?
        hermitMPS[i] = mps_dag_i + mps[i]
    end
    hermitMPS /= 2

    # compute expectation values
    expVals = zeros(Float64, N);
    for i = 1 : N

        # bring MPS into canonical form
        hermitMPS = orthogonalizeMPS(hermitMPS, i); #XXX: Issue 
        psiNormSq = real(tr(hermitMPS[i]' * hermitMPS[i]));
        
        # compute expectation value
        expVal = @tensor conj(hermitMPS[i][-1, 2, 3, -6]) * onsiteOps[i][2, 3, 4, 5] * hermitMPS[i][-1, 4, 5, -6];
        expVals[i] = real(expVal) / psiNormSq;
    end

    return expVals;
end