import LinearAlgebra: kron
import MPSKit: expectation_value
import MPSKitModels: MPOHamiltonian, @mpoham
import TensorKit: TensorMap, isomorphism
include("utils.jl")

### model parameters (See 10.1103/PhysRevLett.123.100604)
L = 7
χ = 128
Ω = 6
γ = 1


basis_0 = convert.(ComplexF64, [1, 0]);
basis_1 = convert.(ComplexF64, [0, 1]);

### MPS 
if isodd(L)
    init_state_MPS = FiniteMPS([fill(TensorMap(basis_0, ℂ^1 ⊗ ℂ^2, ℂ^1), L÷2); 
                                TensorMap(basis_1, ℂ^1 ⊗ ℂ^2, ℂ^1); 
                                fill(TensorMap(basis_0, ℂ^1 ⊗ ℂ^2, ℂ^1), L÷2)]; normalize=true)
else
    init_state_MPS = FiniteMPS([fill(TensorMap(basis_0, ℂ^1 ⊗ ℂ^2, ℂ^1), L÷2 - 1); 
                                TensorMap(basis_1, ℂ^1 ⊗ ℂ^2, ℂ^1); 
                                fill(TensorMap(basis_0, ℂ^1 ⊗ ℂ^2, ℂ^1), L÷2)]; normalize=true)
end

number_op_chain = @mpoham sum(number_op{i} for i in vertices(FiniteChain(L)))

number_density = expectation_value(init_state_MPS, number_op_chain)
av_number_density = abs(sum(number_density))/L
@show av_number_density

### vectorized MPO -> MPS representation

if isodd(L)
    init_state_MPO = FiniteMPS([fill(TensorMap(kron(basis_0, basis_0), ℂ^1 ⊗ ℂ^4, ℂ^1), L÷2); 
                             TensorMap(kron(basis_1, basis_1), ℂ^1 ⊗ ℂ^4, ℂ^1); 
                             fill(TensorMap(kron(basis_0, basis_0), ℂ^1 ⊗ ℂ^4, ℂ^1), L÷2)]; normalize=true)
else
    init_state_MPO = FiniteMPS([fill(TensorMap(kron(basis_0, basis_0), ℂ^1 ⊗ ℂ^4, ℂ^1), L÷2 - 1); 
                             TensorMap(kron(basis_1, basis_1), ℂ^1 ⊗ ℂ^4, ℂ^1); 
                             fill(TensorMap(kron(basis_0, basis_0), ℂ^1 ⊗ ℂ^4, ℂ^1), L÷2)]; normalize=true)
end
# init_state_MPO = FiniteMPS(L, fuse(ℂ^2, conj(ℂ^2)), ℂ^χ) # random state

@show init_state_MPO

## MPO representation of the Lindbladian superoperator
Id_mat = TensorMap(ComplexF64[1 0;0 1], ℂ^2 ← ℂ^2)
fuseIsometry = isomorphism(ℂ^4, ℂ^2 ⊗ ℂ^2) 

sigmax = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
sigmax_r = fuseIsometry * (sigmax ⊗ Id_mat) * fuseIsometry'
sigmax_l = fuseIsometry * (Id_mat ⊗ sigmax) * fuseIsometry'

number_op = TensorMap(ComplexF64[0 0; 0 1], ℂ^2 ← ℂ^2)
number_op_r = fuseIsometry * (number_op ⊗ Id_mat) * fuseIsometry'
number_op_l = fuseIsometry * (Id_mat ⊗ number_op) * fuseIsometry'

annihilation_op = TensorMap(ComplexF64[0 1; 0 0], ℂ^2 ← ℂ^2)
creation_op = TensorMap(ComplexF64[0 0; 1 0], ℂ^2 ← ℂ^2)

on_site = γ * fuseIsometry * (annihilation_op ⊗ annihilation_op) * fuseIsometry' - (1/2) * number_op_r - (1/2) * number_op_l
on_site_dag = γ * fuseIsometry * (creation_op ⊗ creation_op) * fuseIsometry' - (1/2) * number_op_r - (1/2) * number_op_l

Lindbladian = Array{Any, 3}(missing, L, 6, 6)
Lindbladian[1, 1, :] = [1, -1im*Ω*sigmax_r, -1im*sigmax_l, -1im*number_op_r, -1im*number_op_r, on_site] 
for i = 2:L-1
    Lindbladian[i, 1, :] = [1, -1im*Ω*sigmax_r, -1im*sigmax_l, -1im*number_op_r, -1im*number_op_r, on_site]
    Lindbladian[i, 2, 6] = number_op_r
    Lindbladian[i, 3, 6] = number_op_l
    Lindbladian[i, 4, 6] = sigmax_r
    Lindbladian[i, 5, 6] = sigmax_l
    Lindbladian[i, 6, 6] = 1
end
Lindbladian[L, :, 6] = [on_site, number_op_r, number_op_l, sigmax_r, sigmax_l, 1]
Lindbladian = MPOHamiltonian(Lindbladian)

Lindbladian_dag = Array{Any, 3}(missing, L, 6, 6)
Lindbladian_dag[1, 1, :] = [1, im*Ω*sigmax_r, im*sigmax_l, im*number_op_r, im*number_op_r, on_site] 
for i = 2:L-1
    Lindbladian_dag[i, 1, :] = [1, im*Ω*sigmax_r, im*sigmax_l, im*number_op_r, im*number_op_r, on_site]
    Lindbladian_dag[i, 2, 6] = number_op_r
    Lindbladian_dag[i, 3, 6] = number_op_l
    Lindbladian_dag[i, 4, 6] = sigmax_r
    Lindbladian_dag[i, 5, 6] = sigmax_l
    Lindbladian_dag[i, 6, 6] = 1
end
Lindbladian_dag[L, :, 6] = [on_site_dag, number_op_r, number_op_l, sigmax_r, sigmax_l, 1]
Lindbladian_dag = MPOHamiltonian(Lindbladian_dag)

Lindbladian_hermit = Lindbladian_dag*Lindbladian

## run DMRG
groundstate, environment, convergence_err = find_groundstate!(init_state_MPO, Lindbladian_hermit, DMRG2(trscheme=truncbelow(1e-7)))


## calculation of expectation value for MPO of the form (ρ + ρ^†)/2
number_op_l_chain = @mpoham sum(number_op_l{i} for i in vertices(FiniteChain(L)))

altered_ψ, _ = apply_MPO!(groundstate, number_op_l_chain)