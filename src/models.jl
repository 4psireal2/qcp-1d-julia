using LinearAlgebra
using TensorKit

function expCPHam(omega, dt)
    sigmaX = 0.5 * [0 +1; +1 0]
    numberOp = [0 0; 0 1]
    ham = omega * (kron(sigmaX, numberOp) + kron(numberOp, sigmaX))

    HamOp = TensorMap(ham, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
    expHamOp = exp(-1im * dt* HamOp)

    return expHamOp
end

function expCPDiss(gamma, dt)
    annihilationOp = [0 1; 0 0]
    numberOp = [0 0; 0 1]
    Id = [+1 0; 0 +1]
    numberOpR = kron(numberOp, Id)
    numberOpL = kron(Id, numberOp)

    diss =
        gamma * kron(annihilationOp, annihilationOp) - (1 / 2) * numberOpR -
        (1 / 2) * numberOpL
    diss = TensorMap(
        diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
    )
    diss = exp(dt * diss)

    # EVD    
    D, V = eig(diss, (1, 3), (2, 4))
    B = permute(sqrt(D) * V', (3, 1), (2,))

    return B
end

function constructTFIMPO(J, h, N)
    """ Returns TFI MPO for N sites """

    # set Pauli matrices
    Sx = [0 +1; +1 0]
    Sz = [+1 0; 0 -1]
    Id = [+1 0; 0 +1]
    d = 2

    # initialize Heisenberg MPO
    tfiMPO = Vector{TensorMap}(undef, N)

    # left MPO boundary
    HL = zeros(ComplexF64, 1, d, d, 3)
    HL[1, :, :, 1] = Id
    HL[1, :, :, 2] = Sz
    HL[1, :, :, 3] = -h * Sx
    tfiMPO[1] = TensorMap(
        HL, ComplexSpace(1) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(3)
    )

    # bulk MPO
    for idxMPO in 2:(N - 1)
        HC = zeros(ComplexF64, 3, d, d, 3)
        HC[1, :, :, 1] = Id
        HC[1, :, :, 2] = Sz
        HC[1, :, :, 3] = -h * Sx
        HC[2, :, :, 3] = -J * Sz
        HC[3, :, :, 3] = Id
        tfiMPO[idxMPO] = TensorMap(
            HC, ComplexSpace(3) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(3)
        )
    end

    # right MPO boundary
    HR = zeros(ComplexF64, 3, d, d, 1)
    HR[1, :, :, 1] = -h * Sx
    HR[2, :, :, 1] = -J * Sz
    HR[3, :, :, 1] = Id
    tfiMPO[N] = TensorMap(
        HR, ComplexSpace(3) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(1)
    )

    # function return
    return tfiMPO
end

function constructLindbladMPO(omega::Float64, gamma::Float64, N::Int64)::Vector{TensorMap}
    """
    Construct MPO for the Lindbladian of the contact process
    """
    # define operators
    Id = [+1 0; 0 +1]

    sigmaX = 0.5 * [0 +1; +1 0]
    sigmaXR = kron(sigmaX, Id)
    sigmaXL = kron(Id, sigmaX)

    numberOp = [0 0; 0 1]
    numberOpR = kron(numberOp, Id)
    numberOpL = kron(Id, numberOp)

    annihilationOp = [0 1; 0 0]

    onSite =
        gamma * kron(annihilationOp, annihilationOp) - (1 / 2) * numberOpR -
        (1 / 2) * numberOpL

    lindbladMPO = Vector{TensorMap}(undef, N)

    # left MPO boundary
    leftB = zeros(ComplexF64, 1, 4, 4, 6)
    leftB[1, :, :, 1] = kron(Id, Id)
    leftB[1, :, :, 2] = sigmaXR
    leftB[1, :, :, 3] = sigmaXL
    leftB[1, :, :, 4] = numberOpR
    leftB[1, :, :, 5] = numberOpL
    leftB[1, :, :, 6] = gamma * onSite
    lindbladMPO[1] = TensorMap(
        leftB,
        ComplexSpace(1) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(6),
    )

    # bulk MPO
    for i in 2:(N - 1)
        bulk = zeros(ComplexF64, 6, 4, 4, 6)
        bulk[1, :, :, 1] = kron(Id, Id)
        bulk[1, :, :, 2] = sigmaXR
        bulk[1, :, :, 3] = sigmaXL
        bulk[1, :, :, 4] = numberOpR
        bulk[1, :, :, 5] = numberOpL
        bulk[1, :, :, 6] = gamma * onSite
        bulk[2, :, :, 6] = -1im * omega * numberOpR
        bulk[3, :, :, 6] = 1im * omega * numberOpL
        bulk[4, :, :, 6] = -1im * omega * sigmaXR
        bulk[5, :, :, 6] = 1im * omega * sigmaXL
        bulk[6, :, :, 6] = kron(Id, Id)

        lindbladMPO[i] = TensorMap(
            bulk,
            ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
            ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(6),
        )
    end

    # right MPO boundary
    rightB = zeros(ComplexF64, 6, 4, 4, 1)
    rightB[1, :, :, 1] = gamma * onSite
    rightB[2, :, :, 1] = -1im * omega * numberOpR
    rightB[3, :, :, 1] = 1im * omega * numberOpL
    rightB[4, :, :, 1] = -1im * omega * sigmaXR
    rightB[5, :, :, 1] = 1im * omega * sigmaXL
    rightB[6, :, :, 1] = kron(Id, Id)
    lindbladMPO[N] = TensorMap(
        rightB,
        ComplexSpace(6) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2)',
        ComplexSpace(2) ⊗ ComplexSpace(2)' ⊗ ComplexSpace(1),
    )

    return lindbladMPO
end