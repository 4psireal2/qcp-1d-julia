using LinearAlgebra
using TensorKit

function expHam(omega, dt)
    sigmaX = 0.5 * [0 +1; +1 0]
    numberOp = [0 0; 0 1]
    ham = omega * (kron(sigmaX, numberOp) + kron(numberOp, sigmaX))

    propagator = exp(-1im * dt * ham)
    expHamOp = TensorMap(propagator, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)

    return expHamOp
end

function expDiss(gamma, dt)
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
