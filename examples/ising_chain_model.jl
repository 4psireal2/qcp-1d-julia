using LinearAlgebra
using TensorKit


function constructLindbladMPO(gamma::Float64, V::Float64, omega::Float64, delta::Float64, N::Int64)::Vector{TensorMap}
    """
    Construct MPO for Lindbladian of a dissipative Ising chain
    """
    # define operators
    Id = [+1 0 ; 0 +1];

    sigmaX = 0.5 * [0 +1 ; +1 0];
    sigmaXR = kron(sigmaX, Id);
    sigmaXL = kron(Id, sigmaX);

    sigmaZ = 0.5 * [+1 0 ; 0 -1];
    sigmaZR = kron(sigmaZ, Id);
    sigmaZL = kron(Id, sigmaZ);

    annihilationOp = [0 1; 0 0];
    creationOp = [0 0; 1 0];
    anOp = annihilationOp*creationOp;
    anOpR = kron(anOp, Id);
    anOpL = kron(Id, anOp);
    
    numberOp = [0 0; 0 1];
    numberOpR = kron(numberOp, Id);
    numberOpL = kron(Id, numberOp);


    onSite = -1im*(omega/2)*(sigmaXL + sigmaXR) + 1im*(V-delta)/2*(sigmaZL+sigmaZR) + gamma*kron(creationOp, creationOp) - (1/2)*gamma*(anOpL + anOpR);
    # onSite = -1im*(omega/2)*(sigmaXL + sigmaXR) + 1im*(V-delta)/2*(sigmaZL+sigmaZR) + gamma*kron(annihilationOp, annihilationOp) - (1/2)*gamma*(numberOpR + numberOpL);

    lindbladMPO = Vector{TensorMap}(undef, N);

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 4)
    leftB[1, :, :, 1] = kron(Id, Id);
    leftB[1, :, :, 2] = sigmaZR;
    leftB[1, :, :, 3] = sigmaZL;
    leftB[1, :, :, 4] = onSite + (V/4)*(sigmaZL + sigmaZR);
    lindbladMPO[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(4));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 4, 4, 4, 4);
        bulk[1, :, :, 1] = kron(Id, Id);
        bulk[1, :, :, 2] = sigmaZR;
        bulk[1, :, :, 3] = sigmaZL;
        bulk[1, :, :, 4] = onSite;
        bulk[2, :, :, 4] = -1im * V/4 * sigmaZR;
        bulk[3, :, :, 4] = -1im * V/4 * sigmaZL;
        bulk[4, :, :, 4] = kron(Id, Id);        
        lindbladMPO[i] = TensorMap(bulk, ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(4));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 4, 4, 4, 1)
    rightB[1, :, :, 1] = onSite + (V/4)*(sigmaZL + sigmaZR);
    rightB[2, :, :, 1] = -1im * V/4 * sigmaZR;
    rightB[3, :, :, 1] = -1im * V/4 * sigmaZL;
    rightB[4, :, :, 1] = kron(Id, Id);
    lindbladMPO[N] = TensorMap(rightB, ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    return lindbladMPO;
end


function constructLindbladDagMPO(gamma::Float64, V::Float64, omega::Float64, delta::Float64, N::Int64)::Vector{TensorMap}
    """
    Construct MPO for adjoint Lindbladian of a dissipative Ising chain
    """
    # define operators
    Id = [+1 0 ; 0 +1];

    sigmaX = 0.5 * [0 +1 ; +1 0];
    sigmaXR = kron(sigmaX, Id);
    sigmaXL = kron(Id, sigmaX);

    sigmaZ = 0.5 * [+1 0 ; 0 -1];
    sigmaZR = kron(sigmaZ, Id);
    sigmaZL = kron(Id, sigmaZ);

    annihilationOp = [0 1; 0 0];
    creationOp = [0 0; 1 0];
    anOp = annihilationOp*creationOp;
    anOpR = kron(anOp, Id);
    anOpL = kron(Id, anOp);

    numberOp = [0 0; 0 1];
    numberOpR = kron(numberOp, Id);
    numberOpL = kron(Id, numberOp);

    onSite = 1im*(omega/2)*(sigmaXL + sigmaXR) - 1im*(V-delta)/2*(sigmaZL+sigmaZR) + gamma*kron(annihilationOp, annihilationOp) - (1/2)*gamma*(anOpL + anOpR);
    # onSite = 1im*(omega/2)*(sigmaXL + sigmaXR) - 1im*(V-delta)/2*(sigmaZL+sigmaZR) + gamma*kron(creationOp, creationOp) - (1/2)*gamma*(numberOpR + numberOpL);

    lindbladMPO = Vector{TensorMap}(undef, N);

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 4)
    leftB[1, :, :, 1] = kron(Id, Id);
    leftB[1, :, :, 2] = sigmaZR;
    leftB[1, :, :, 3] = sigmaZL;
    leftB[1, :, :, 4] = onSite + (V/4)*(sigmaZL + sigmaZR);
    lindbladMPO[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(4));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 4, 4, 4, 4);
        bulk[1, :, :, 1] = kron(Id, Id);
        bulk[1, :, :, 2] = sigmaZR;
        bulk[1, :, :, 3] = sigmaZL;
        bulk[1, :, :, 4] = onSite;
        bulk[2, :, :, 4] = 1im * V/4 * sigmaZR;
        bulk[3, :, :, 4] = 1im * V/4 * sigmaZL;
        bulk[4, :, :, 4] = kron(Id, Id);        
        lindbladMPO[i] = TensorMap(bulk, ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(4));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 4, 4, 4, 1)
    rightB[1, :, :, 1] = onSite + (V/4)*(sigmaZL + sigmaZR);
    rightB[2, :, :, 1] = 1im * V/4 * sigmaZR;
    rightB[3, :, :, 1] = 1im * V/4 * sigmaZL;
    rightB[4, :, :, 1] = kron(Id, Id);
    lindbladMPO[N] = TensorMap(rightB, ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));
    
    return lindbladMPO;
end


function constructMzStags(N::Int64)::Vector{TensorMap}
    """
    Construct MPO for the staggered magnetisation operator
    """

    mpo = Vector{TensorMap}(undef, N);
    Mz = 0.5 * (kron([+1 0 ; 0 -1], [1 0; 0 1]) + kron([1 0; 0 1], [+1 0 ; 0 -1]));

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 1);
    leftB[1, :, :, 1]  = (-1im)^1 * Mz;
    mpo[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 1, 4, 4, 1);
        bulk[1, :, :, 1]  = (-1im)^i * Mz;
        mpo[i] = TensorMap(bulk, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 1, 4, 4, 1);
    rightB[1, :, :, 1]  = (-1im)^N * Mz;
    mpo[N] = TensorMap(rightB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    return mpo/N;
end


function constructMzStagsDag(N::Int64)::Vector{TensorMap}
    """
    Construct adjoint MPO for the staggered magnetisation operator
    """

    mpo = Vector{TensorMap}(undef, N);
    Mz = 0.5 * (kron([+1 0 ; 0 -1], [1 0; 0 1]) + kron([1 0; 0 1], [+1 0 ; 0 -1]));

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 1);
    leftB[1, :, :, 1]  = (1im)^1 * Mz;
    mpo[1] = TensorMap(leftB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    # bulk MPO
    for i = 2 : (N-1)
        bulk = zeros(ComplexF64, 1, 4, 4, 1);
        bulk[1, :, :, 1]  = (1im)^i * Mz;
        mpo[i] = TensorMap(bulk, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 1, 4, 4, 1);
    rightB[1, :, :, 1]  = (1im)^N * Mz;    
    mpo[N] = TensorMap(rightB, ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2) ⊗ ComplexSpace(2) ⊗ ComplexSpace(1));

    return mpo/N;
end