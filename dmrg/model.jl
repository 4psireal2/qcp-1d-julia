using MPSKit, MPSKitModels, TensorKit, Plots

L = 6
χ = 4
Ω = 6
γ = 1

init_state_MPS = FiniteMPS(L, ℂ^2, ℂ^χ) 
println("MPS state: $init_state_MPS")

basis_0 = TensorMap([1, 0], ℂ^2 ← ℂ^1)
basis_1 = TensorMap([0, 1], ℂ^2 ← ℂ^1)

init_product_state = basis_0 ⊗ basis_0 ⊗basis_0 ⊗ basis_1 ⊗ basis_1 ⊗ basis_1 
# TensorMap((ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2) ← (ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1))
#TODO: Perform a SVD such that init_product_state = init_state_MPS

init_state_MPO = FiniteMPS(L, ℂ^2 ⊗ conj(ℂ^2), ℂ^χ) # hmm dim is already represented as product of dims
println("Vectorized-MPO state: $init_state_MPO")

# MPO representation of the Lindbladian superoperator
T = ComplexF64
Id_mat = TensorMap(T[1 0;0 1], ℂ^2 ← ℂ^2)

sigmax = TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
sigmax_r = sigmax ⊗ Id_mat
sigmax_l = Id_mat ⊗ sigmax

number_op = TensorMap(T[0 0; 0 1], ℂ^2 ← ℂ^2)
number_op_r = number_op ⊗ Id_mat
number_op_l = Id_mat ⊗ number_op

annihilation_op = TensorMap(T[0 1; 0 0], ℂ^2 ← ℂ^2)
creation_op = TensorMap(T[0 0; 1 0], ℂ^2 ← ℂ^2)

on_site = γ * (annihilation_op ⊗ creation_op - (1/2) * number_op_r - (1/2) * number_op_l)

Linbladian = Array{Any, 3}(missing, L, 6, 6)
Linbladian[1, 1, :] = [1, -1im*Ω*sigmax_r, -1im*sigmax_l, -1im*number_op_r, -1im*number_op_r, on_site] 
for i = 2:L-1
    Linbladian[i, 1, :] = [1, -1im*Ω*sigmax_r, -1im*sigmax_l, -1im*number_op_r, -1im*number_op_r, on_site]
    # Linbladian[i, 2, 6] = transpose(number_op_r)
    # Linbladian[i, 3, 6] = transpose(number_op_l)
    # Linbladian[i, 4, 6] = transpose(sigmax_r)
    # Linbladian[i, 5, 6] = transpose(sigmax_l)
    Linbladian[i, 2, 6] = number_op_r
    Linbladian[i, 3, 6] = number_op_l
    Linbladian[i, 4, 6] = sigmax_r
    Linbladian[i, 5, 6] = sigmax_l
    Linbladian[i, 6, 6] = 1
end
Linbladian[L, :, 6] = [on_site, missing, missing, missing, missing, 1]
Linbladian = MPOHamiltonian(Linbladian)

# MPO representation of the particle number operator
number_op_chain = @mpoham sum(number_op{i} for i in vertices(FiniteChain(L)))

number_density = expectation_value(init_state_MPS, number_op_chain)
av_number_density = abs(sum(number_density))/L