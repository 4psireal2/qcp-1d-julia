using LinearAlgebra
using TensorKit


function constructLindbladMPO(omega::Float64, gamma::Float64, N::Int64)::Vector{TensorMap}
    """
    Construct MPO for Lindbladian of a contact process
    """
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
    Construct MPO for adjoint Lindbladian of a contact process
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


function constructNumberOps(N::Int64)::Vector{TensorMap}
    """
    Construct particle number operator
    """
    numberOps = Vector{TensorMap}(undef, N);
    numberOp = (kron([0 0; 0 1], [1 0; 0 1]) + kron([1 0; 0 1], [0 0; 0 1]));
    
    for i = 1 : N
        numberOps[i] = TensorMap(numberOp, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
    end

    return numberOps;