using TensorKit
using LinearAlgebra

function randHaar(dim::Int)
    A = randn(ComplexF64, dim, dim)
    Q, R = qr(A)
    Q *= Diagonal(sign.(diag(R)))

    return Q
end

function depolarizingNoise(p)
    """
    Ref: DOI: 10.1103/PhysRevResearch.3.023005 [Eq. 2.5]

    Params:
    - 0 <= p <= 1
    """
    Sx = [0 +1; +1 0]
    Sy = [0 -1im; +1im 0]
    Sz = [+1 0; 0 -1]
    haarMat = randHaar(2)

    noise =
        (1 - p) * kron(haarMat, conj(haarMat)) +
        p / 2 * (kron(Sx, conj(Sx)) + kron(Sy, conj(Sy)) + kron(Sz, conj(Sz)))
    noise = TensorMap(
        noise, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
    )

    # EVD    
    D, V = eig(noise, (1, 3), (2, 4))
    B = permute(sqrt(D) * V', (3, 1), (2,))

    return B
end
