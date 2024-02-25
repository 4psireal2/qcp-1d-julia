""" Build a tranverse field Ising Hamiltonian
Hamiltonian: -J(∑_{i,j} σ^x_i  σ^x_j + g∑_i σ^z_i)   
Guide: https://github.com/maartenvd/MPSKit.jl/blob/45cab94fb3d78169a5987c54ed2b762652dd4c70/docs/src/man/operators.md """

using MPSKit, MPSKitModels, TensorKit
using ProgressMeter, Plots
using LinearAlgebra


L = 16
D = 128


T = ComplexF64
X = TensorMap(T[0 1;1 0], ℂ^2 ← ℂ^2)
Z = TensorMap(T[1 0;0 -1], ℂ^2 ← ℂ^2)
Id_mat = TensorMap(T[1 0;0 1], ℂ^2 ← ℂ^2)
J = 1.0
g = 1.0


init_state = FiniteMPS(L, ℂ^2, ℂ^D)


## L-site Hamiltonian
data = Array{Any,3}(missing,L,3,3)
data[1, 1, 1] = Id_mat
data[1, 1, 2] = X
data[1, 1, 3] = -J*g*Z
for i = 2:L-1
	data[i, 1, 1]= Id_mat
	data[i, 1, 2] = -J * X
	data[i, 1, 3] = -J * g * Z
	data[i, 2, 3] = -J * X
	data[i, 3, 3] = Id_mat
end
data[L, 1, 1] = -J*g*Z
data[L, 2, 1] = X
data[L, 3, 1] = Id_mat

## OBC
println("Computation 1")
OBC_Ising = @mpoham sum(-J * S_xx(){i, j} -J * g * S_z(){k} for (i,j) in nearest_neighbours(FiniteChain(L)) for k in vertices(FiniteChain(L)))
ψ_obc, env_obc, δ_obc = find_groundstate(init_state, OBC_Ising; verbose=true) #XXX: strange energy


println("Computation 2")
OBC_Ising = MPOHamiltonian(data)
ψ_obc_own, env_obc_own, δ_obc_own = find_groundstate(init_state, OBC_Ising; verbose=true) #XXX: strange energy, ϵ = NaN

## 1-site Hamiltonian
data = Array{Any,3}(missing, 1, 3, 3)
data[1, 1, 1] = Id_mat
data[1, 1, 2] = -J * X
data[1, 1, 3] = -J * g * Z
data[1, 2, 3] = -J * X
data[1, 3, 3] = Id_mat

## PBC
println("Computation 3")
PBC_Ising = periodic_boundary_conditions(MPOHamiltonian(data), L)
ψ_pbc_own, env_pbc_own, δ_pbc_own = find_groundstate(init_state, PBC_Ising; verbose=true) #XXX: strange energy, ϵ = NaN


println("Computation 4")
PBC_Ising = periodic_boundary_conditions(transverse_field_ising(; g=g), L)
ψ_pbc, env_pbc, δ_pbc = find_groundstate(init_state, PBC_Ising; verbose=true) #XXX: strange energy

println("Literature energy E/$L: -1.2510242438)") #Ref: https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
println("OBC Ising own computation: $δ_obc_own") # δ=2.0e-12
println("OBC Ising MPSKit computation: $δ_obc") # δ=2.0e-12
println("PBC Ising own computation: $δ_pbc_own") # δ=2.0e-12
println("PBC Ising MPSKit computation: $δ_pbc") # δ=2.0e-12