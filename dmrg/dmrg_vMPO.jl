using LinearAlgebra
import LinearAlgebra: norm, dot, tr, diag, diagm

using KrylovKit
using Printf
using TensorKit
import KrylovKit: eigsolve
import TensorKit: leftorth, randn, truncdim, ComplexSpace


function initializeRandomVMPO(N, d::Int64 = 2, bonddim:: Int64 = 1)::Vector
    """
    Return random vectorized MPO (MPS-like) for N sites
    """

    randomVMPO = Vector{TensorMap}(undef, N);

    # left MPO boundary
    randomVMPO[1] = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(bonddim));
    for i = 2 : (N-1)
        randomVMPO[i] = TensorMap(randn, ComplexSpace(bonddim) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(bonddim));
    end
    randomVMPO[N] = TensorMap(randn, ComplexSpace(bonddim) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(1));

    return randomVMPO;
end


function orthogonalizeVMPO(fVMPO::Vector{<:AbstractTensorMap}, orthoCenter::Int)
    N = length(fVMPO);
    orthoVMPO = deepcopy(fVMPO);

    # bring sites 1 to orthoCenter-1 into left-orthogonal form
    for i = 1 : 1 : (orthoCenter - 1)
        Q, R = leftorth(orthoVMPO[i], (1, 2, 3), (4, ), alg = QRpos());

        orthoVMPO[i + 0] = permute(Q, (1,2), (3, 4)) ;
        orthoVMPO[i + 1] = permute(R * permute(orthoVMPO[i + 1], (1, ), (2, 3, 4)), (1, 2), (3, 4))
    end

    # bring sites orthCenter + 1 to N into right-canonical form
    for i = N : -1 : (orthoCenter + 1)
        L, Q = rightorth(orthoVMPO[i], (1, ), (2, 3, 4), alg = LQpos());

        orthoVMPO[i - 1] = permute(permute(orthoVMPO[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
        orthoVMPO[i - 0] = permute(Q, (1, 2), (3, 4));
    end
    return orthoVMPO;
end


function orthonormalizeVMPO(fVMPO::Vector{<:AbstractTensorMap})
    orthoVMPO = orthogonalizeVMPO(fVMPO, 1);
    normVMPO = real(tr(orthoVMPO[1]' * orthoVMPO[1]));
    orthoVMPO[1] /= sqrt(normVMPO);
    return orthoVMPO;
end


function constructLindbladMPO(omega, gamma, N)
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


function constructLindbladDagMPO(omega, gamma, N)
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

function multiplyMPOMPO(fMPO1, fMPO2)


        




finiteVMPO = initializeRandomMPO(4);
orthonormalizeVMPO(finiteVMPO);
