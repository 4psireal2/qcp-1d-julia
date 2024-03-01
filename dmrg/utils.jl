"""Contain utility functions"""


import KrylovKit: Arnoldi, eigsolve
import MPSKit: ∂∂AC, environments, FiniteMPS
import MPSKitModels: MPOHamiltonian


function apply_MPO(ψ::FiniteMPS, H::MPOHamiltonian, envs=environments(ψ, H))
    tol = 1e-12
    maxiter = 100 

    for pos in [1:(length(ψ) - 1); length(ψ):-1:2] # sweeping forward and backward?
        h = ∂∂AC(pos, ψ, H, envs)
        _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, Arnoldi(; tol, maxiter, eager=true))
        ψ.AC[pos] = vecs[1]
    end

    return ψ, envs
end


function find_trace(ψ::FiniteMPS, envs)
    #TODO: write function
end