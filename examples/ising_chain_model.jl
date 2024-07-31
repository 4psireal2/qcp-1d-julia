using LinearAlgebra
using Printf
using KrylovKit
using MPSKit
using TensorKit

function constructLindbladMPO(
    gamma::Float64, V::Float64, omega::Float64, delta::Float64, N::Int64
)::Vector{TensorMap}
    """
    Construct MPO for Lindbladian of a dissipative Ising chain
    """
    # define operators
    Id = [+1 0; 0 +1]

    sigmaX = 0.5 * [0 +1; +1 0]
    sigmaXL = kron(sigmaX, Id)
    sigmaXR = kron(Id, sigmaX)

    sigmaZ = 0.5 * [+1 0; 0 -1]
    sigmaZL = kron(sigmaZ, Id)
    sigmaZR = kron(Id, sigmaZ)

    annihilationOp = [0 1; 0 0]
    creationOp = [0 0; 1 0]
    anOp = annihilationOp * creationOp

    anOpL = kron(anOp, Id)
    anOpR = kron(Id, anOp)

    onSite =
        -1im * (omega / 2) * (sigmaXL - sigmaXR) +
        1im * (V - delta) / 2 * (sigmaZL - sigmaZR) +
        gamma * kron(creationOp, creationOp) - (1 / 2) * gamma * (anOpL + anOpR)

    lindbladMPO = Vector{TensorMap}(undef, N)

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 4)
    leftB[1, :, :, 1] = kron(Id, Id)
    leftB[1, :, :, 2] = sigmaZR
    leftB[1, :, :, 3] = sigmaZL
    leftB[1, :, :, 4] = onSite + -1im * (V / 4) * (sigmaZL - sigmaZR)
    lindbladMPO[1] = TensorMap(
        leftB,
        ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(4),
    )

    # bulk MPO
    for i in 2:(N - 1)
        bulk = zeros(ComplexF64, 4, 4, 4, 4)
        bulk[1, :, :, 1] = kron(Id, Id)
        bulk[1, :, :, 2] = sigmaZR
        bulk[1, :, :, 3] = sigmaZL
        bulk[1, :, :, 4] = onSite
        bulk[2, :, :, 4] = 1im * V / 4 * sigmaZR
        bulk[3, :, :, 4] = -1im * V / 4 * sigmaZL
        bulk[4, :, :, 4] = kron(Id, Id)
        lindbladMPO[i] = TensorMap(
            bulk,
            ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
            ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(4),
        )
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 4, 4, 4, 1)
    rightB[1, :, :, 1] = onSite + -1im * (V / 4) * (sigmaZL - sigmaZR)
    rightB[2, :, :, 1] = 1im * V / 4 * sigmaZR
    rightB[3, :, :, 1] = -1im * V / 4 * sigmaZL
    rightB[4, :, :, 1] = kron(Id, Id)
    lindbladMPO[N] = TensorMap(
        rightB,
        ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(1),
    )

    return lindbladMPO
end

function constructLindbladDagMPO(
    gamma::Float64, V::Float64, omega::Float64, delta::Float64, N::Int64
)::Vector{TensorMap}
    """
    Construct MPO for Lindbladian of a dissipative Ising chain
    """
    # define operators
    Id = [+1 0; 0 +1]

    sigmaX = 0.5 * [0 +1; +1 0]
    sigmaXL = kron(sigmaX, Id)
    sigmaXR = kron(Id, sigmaX)

    sigmaZ = 0.5 * [+1 0; 0 -1]
    sigmaZL = kron(sigmaZ, Id)
    sigmaZR = kron(Id, sigmaZ)

    annihilationOp = [0 1; 0 0]
    creationOp = [0 0; 1 0]
    anOp = annihilationOp * creationOp

    anOpL = kron(anOp, Id)
    anOpR = kron(Id, anOp)

    onSite =
        1im * (omega / 2) * (sigmaXL - sigmaXR) -
        1im * (V - delta) / 2 * (sigmaZL - sigmaZR) +
        gamma * kron(annihilationOp, annihilationOp) - (1 / 2) * gamma * (anOpL + anOpR)

    lindbladMPO = Vector{TensorMap}(undef, N)

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 4)
    leftB[1, :, :, 1] = kron(Id, Id)
    leftB[1, :, :, 2] = sigmaZR
    leftB[1, :, :, 3] = sigmaZL
    leftB[1, :, :, 4] = onSite + 1im * (V / 4) * (sigmaZL - sigmaZR)
    lindbladMPO[1] = TensorMap(
        leftB,
        ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(4),
    )

    # bulk MPO
    for i in 2:(N - 1)
        bulk = zeros(ComplexF64, 4, 4, 4, 4)
        bulk[1, :, :, 1] = kron(Id, Id)
        bulk[1, :, :, 2] = sigmaZR
        bulk[1, :, :, 3] = sigmaZL
        bulk[1, :, :, 4] = onSite
        bulk[2, :, :, 4] = -1im * V / 4 * sigmaZR
        bulk[3, :, :, 4] = 1im * V / 4 * sigmaZL
        bulk[4, :, :, 4] = kron(Id, Id)
        lindbladMPO[i] = TensorMap(
            bulk,
            ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
            ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(4),
        )
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 4, 4, 4, 1)
    rightB[1, :, :, 1] = onSite + 1im * (V / 4) * (sigmaZL - sigmaZR)
    rightB[2, :, :, 1] = -1im * V / 4 * sigmaZR
    rightB[3, :, :, 1] = 1im * V / 4 * sigmaZL
    rightB[4, :, :, 1] = kron(Id, Id)
    lindbladMPO[N] = TensorMap(
        rightB,
        ComplexSpace(4) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(1),
    )

    return lindbladMPO
end
