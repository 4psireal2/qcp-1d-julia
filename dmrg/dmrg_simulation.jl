using TensorKit
using Profile

include("dmrg_vMPO.jl")


N = 20;
d = 2;
bondDim = 1;

### basis states
basis_0 = [1, 0];
basis_1 = [0, 1];

# initialize basis MPS
basis = fill(basis_1, N);
initialMPS = initializeBasisMPS(N, basis, d=d);
initialMPS = orthonormalizeMPS(initialMPS);


# initialize random MPS
# initialMPS = initializeRandomMPS(N);
# initialMPS = orthonormalizeMPS(initialMPS);

# construct Lindbladian MPO for quantum contact process
lindblad1 = constructLindbladMPO(6.0, 1.0, N);
lindblad2 = constructLindbladDagMPO(6.0, 1.0, N);
lindbladHermitian = multiplyMPOMPO(lindblad1, lindblad2);

# run DMRG
# elapsed_time = @elapsed begin
#     gsMPS, gsEnergy = DMRG2(initialMPS, lindbladHermitian, bondDim = 16, truncErr = 1e-6, convTolE = 1e-6, maxIterations=10, verbosePrint = true);
# end
# println("Elapsed time for DMRG2: $elapsed_time seconds")
# @sprintf("ground state energy per site E = %0.6f", gsEnergy / N)

# compute particle numbers
numberOps = Vector{TensorMap}(undef, N);
numberOp = kron([0 0; 0 1], [0 0; 0 1]);
for i = 1 : N
    numberOps[i] = TensorMap(numberOp, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2);
end

particleNums = computeSiteExpVal(initialMPS, numberOps); ###XXX: Does it make sense to have max=0.5 for each site?

### Debugging
# @tensor test1[-1 -2 -3; -4 -5 -6] := lindblad1[1][1 2 3; -4 -5 4] * lindblad2[1][5 -2 -3; 2 3 6] * fusers[1][-1; 1 5] * conj(fusers[2][-6; 4 6]);
# @tensor test2[-1 -2; -3 -4 -5 -6] :=  lindblad1[1][-2, 2, 3, -3, -4, -5] * conj(initialMPS[1][-1, 2, 3, -6]);

## Check for Hermiticity
# N = 3;
# physicalDim = 2;

# # initialize MPS
# initialMPS = initializeRandomMPS(N);
# orthonormalizeMPS(initialMPS);

# # construct Lindbladian MPO for quantum contact process
# lindblad1 = constructLindbladMPO(6.0, 1.0, N);
# lindblad2 = constructLindbladDagMPO(6.0, 1.0, N);

# boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
# boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
# lindbladHermitian = multiplyMPOMPO(lindblad2, lindblad1);
# @tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := boundaryL[1] * lindbladHermitian[1][1, -1, -2, -7, -8, 2] *
#                                                                                lindbladHermitian[2][2, -3, -4, -9, -10, 3] *
#                                                                                lindbladHermitian[3][3, -5, -6, -11, -12, 4] *
#                                                                                boundaryR[4];

# lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));


# norm(lindbladMatrix - lindbladMatrix') == 0.0
# lindbladHermitianDag = lindbladHermitian'

# lindbladHermitian = reshape(lindbladHermitian, (N, ))
# lindbladHermitianDag_permuted = deepcopy(lindbladHermitianDag);
# lindbladHermitianDag_permuted[1] = permute(lindbladHermitianDag[1], (4, 5, 6), (1, 2, 3));

# lindbladHermitianDag_1 =   TensorMap(conj(permutedims(convert(Array, lindbladHermitian[1]), (1, 4, 5, 2, 3, 6))), codomain(lindbladHermitian[1]), domain(lindbladHermitian[1]));
# @show (norm(lindbladHermitian[1] - lindbladHermitianDag_1))

# lindbladHermitianDag_2 =   TensorMap(conj(permutedims(convert(Array, lindbladHermitian[2]), (1, 4, 5, 2, 3, 6))), codomain(lindbladHermitian[2]), domain(lindbladHermitian[2]));
# @show (norm(lindbladHermitian[2] - lindbladHermitianDag_2))

# lindbladHermitianDag_3 =   TensorMap(conj(permutedims(convert(Array, lindbladHermitian[3]), (1, 4, 5, 2, 3, 6))), codomain(lindbladHermitian[3]), domain(lindbladHermitian[3]));
# @show (norm(lindbladHermitian[3] - lindbladHermitianDag_3))

# @tensor test1[-1; 2 3 -6] := numberOps[1][2, 3, 4, 5]*initialMPS[1][-1, 4, 5, -6]
# expVal1 = @tensor conj(initialMPS[1][-1, 2, 3, -6]) * numberOps[1][2, 3, 4, 5] * initialMPS[1][-1, 4, 5, -6]

