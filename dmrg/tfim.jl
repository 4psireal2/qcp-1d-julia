""" Build a tranverse field Ising Hamiltonian
Hamiltonian: -J(∑_{i,j} σ^x_i  σ^x_j + g∑_i σ^z_i)   
Guide: https://github.com/maartenvd/MPSKit.jl/blob/45cab94fb3d78169a5987c54ed2b762652dd4c70/docs/src/man/operators.md """

using MPSKit, MPSKitModels, TensorKit
using ProgressMeter, Plots
using LinearAlgebra
using QuadGK


L = 16
D = 128


T = ComplexF64
X = TensorMap(T[0 1;1 0], ℂ^2 ← ℂ^2)
Z = TensorMap(T[1 0;0 -1], ℂ^2 ← ℂ^2)
Id_mat = TensorMap(T[1 0;0 1], ℂ^2 ← ℂ^2)
J = 1.0
g = 1.0


init_state = FiniteMPS(L, ℂ^2, ℂ^D)
h=1
E_GS = -quadgk(k -> sqrt(1 + h^2 - 2*h*cos(k)), -pi, pi)[1]/(2*pi)


## L-site Hamiltonian
data = Array{Any,3}(missing,L,3,3)
data[1, 1, 1] = Id_mat
data[1, 1, 2] = X
data[1, 1, 3] = -J*g*Z
for i = 2:L-1
	data[i, 1, 1]= Id_mat
	data[i, 1, 2] = -J * X
	data[i, 1, 3] = -J * g * Z
	data[i, 2, 3] = X
	data[i, 3, 3] = Id_mat
end
data[L, 1, 1] = -J*g*Z
data[L, 2, 1] = X
data[L, 3, 1] = Id_mat

## OBC
# println("Computation 1")
# OBC_Ising = @mpoham sum(-J * S_xx(){i, j} -J * g * S_z(){k} for (i,j) in nearest_neighbours(FiniteChain(L)) for k in vertices(FiniteChain(L)))
# ψ_obc, env_obc, δ_obc = find_groundstate(init_state, OBC_Ising; verbose=true) #XXX: strange energy


println("Computation 2")
OBC_Ising = MPOHamiltonian(data)
ψ_obc_own, env_obc_own, δ_obc_own = find_groundstate(init_state, OBC_Ising; verbose=true) #XXX: strange energy, ϵ = NaN
E_density_ising = expectation_value(ψ_obc_own, OBC_Ising, env_obc_own) 


# ## 1-site Hamiltonian
# data = Array{Any,3}(missing, 1, 3, 3)
# data[1, 1, 1] = Id_mat
# data[1, 1, 2] = -J * X
# data[1, 1, 3] = -J * g * Z
# data[1, 2, 3] = X
# data[1, 3, 3] = Id_mat

# ## PBC
# println("Computation 3")
# PBC_Ising = periodic_boundary_conditions(MPOHamiltonian(data), L)
# ψ_pbc_own, env_pbc_own, δ_pbc_own = find_groundstate(init_state, PBC_Ising; verbose=true) #XXX: strange energy, ϵ = NaN


# println("Computation 4")
# PBC_Ising = periodic_boundary_conditions(transverse_field_ising(; g=g), L)
# ψ_pbc, env_pbc, δ_pbc = find_groundstate(init_state, PBC_Ising; verbose=true) #XXX: strange energy

println("Literature energy E/$L: -1.2510242438)") #Ref: https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
@show E_density_ising.-E_GS #All right this matches well in the bulk ! Near the edges we see some deviation which is to be expected on a finite lattice with OBC...

### From developers

#first a note on the GS energy of the ising model : https://theory.leeds.ac.uk/interaction-distance/applications/ising/map-to-free/
#H = sum_i -σ^x_i σ^x_{i+1} + hσ^z_i which in the thermodynamic limit can be diagonalized by a Jordan-Wigner transformation to a quadratic fermionic Hamiltonian.
#The dispersion relation of this Hamiltonian is E(k) = ±sqrt(1 + h^2 - 2hcos(k)) which is critical when g = 1.
#At h=1 the GS energy is -int_0^pi dk E(k) which is :
h = 1 #critical point
E_GS = -quadgk(k -> sqrt(1 + h^2 - 2*h*cos(k)), -pi, pi)[1]/(2*pi)

#Let us define a finite lattice
L = 100
lattice = FiniteChain(L)    
#Let us define an initial state with bond dimension D on this finite lattice :
D = 10
init_state = FiniteMPS(L, ℂ^2, ℂ^D)

#1) use @mpoham to define the Hamiltonian on this finite lattice
NN_part = @mpoham sum(-4*S_xx(){i,j} for (i,j) in nearest_neighbours(lattice)    )    #note the 4*S_xx ! This is because S_xx is the spin matrices but if you want the critical point of Ising to appear at h = J = 1 we need to work with Pauli matrices !
on_site = @mpoham sum(+h*2*S_z(){i} for i in vertices(lattice)) #same comment !
OBC_Ising = NN_part + on_site #add the two MPO hamiltonians. Surely you could have used a one-liner to make this but I'm not sure it would have been clearer...

#2) manually construct the Hamiltonian
#First we will need the Pauli matrices 
T = ComplexF64
X = TensorMap(T[0 1;1 0], ℂ^2 ← ℂ^2)
Z = TensorMap(T[1 0;0 -1], ℂ^2 ← ℂ^2)

##The finite state machine representation of the Ising Hamiltonian is :
data = Array{Any,3}(missing,L,3,3)
data[1, :, :] = [1 -X h*Z ; missing missing missing ; missing missing missing] 
for i = 2:L-1
	data[i, :, :]= [1 -X h*Z ; missing missing X ; missing missing 1]
end
data[L, :, :] = [missing missing h*Z ; missing missing missing ; missing missing 1]
#from this we get : 
OBC_Ising_manual = MPOHamiltonian(data)

#Let us now calculate the GSs for these things :
ψ, env, δ = find_groundstate(init_state, OBC_Ising; verbose=true) #ψ, env, δ are respectively the groundstate, the environment and the convergence error of the groundstate search
ψ_manual, env_manual, δ_manual = find_groundstate(init_state, OBC_Ising_manual; verbose=true) #ψ, env, δ are respectively the groundstate, the environment and the convergence error of the groundstate search

#and finally, let us compare the aquired energies with those of the analytical solution :
E_density_ising = expectation_value(ψ, OBC_Ising, env) 
@show E_density_ising.-E_GS #All right this matches well in the bulk ! Near the edges we see some deviation which is to be expected on a finite lattice with OBC...

E_density_ising_manual = expectation_value(ψ_manual, OBC_Ising_manual, env_manual)
@show E_density_ising_manual.-E_GS #All right this matches well in the bulk ! Near the edges we see some deviation which is to be expected on a finite lattice with OBC...