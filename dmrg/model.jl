using MPSKit, MPSKitModels, TensorKit, Plots

L = 6
χ = 4
Ω = 6
γ = 1

init_state_MPS = FiniteMPS(L, ℂ^2, ℂ^χ)
println("MPS state: $init_state_MPS")

# data = fill(TensorMap(rand, ComplexF64,ℂ^4*ℂ^2, ℂ^2), L) # TensorMap(initializer, scalartype, codomain, domain)
# init_state_data = FiniteMPS(data)
# println("MPS state data: $init_state_data")

init_state_MPO = FiniteMPS(L, ℂ^2 ⊗ conj(ℂ^2), ℂ^χ) # hmm dim is already represented as product of dims
println("Vectorized-MPO state: $init_state_MPO")

# MPO representation of the Lindbladian superoperator
T = ComplexF64
sigmax = TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
annihilation_op = TensorMap(T[0 1; 0 0], ℂ^2 ← ℂ^2)
creation_op = TensorMap(T[0 0; 1 0], ℂ^2 ← ℂ^2)
number_op = TensorMap(T[0 0; 0 1], ℂ^2 ← ℂ^2)

data = Array{Any, 4}(missing, 1, 4, 4, 4) #TODO: dimensions?
data[1,1,1,1] = identity(ℂ^2)
data[1,2,2,1] = Ω * number_op
data[1,3,3,1] = Ω * sigmax
