include("../examples/ising_chain_model.jl")
include("../src/dmrgVMPO.jl")
include("../src/utility.jl")

using LinearAlgebra

L = 3;
GAMMA = 1.0;
OMEGA = 1.5;
V = 5.0;
DELTA = -4.0;

lindblad = constructLindbladMPO(GAMMA, V, OMEGA, DELTA, L);
lindblad_dag = constructLindbladDagMPO(GAMMA, V, OMEGA, DELTA, L);
lindlad_hermitian = multiplyMPOMPO(lindblad, lindblad_dag);
boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))

@tensor lindbladCheck[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] :=
    boundaryL[1] *
    lindlad_hermitian[1][1, -1, -2, -7, -8, 2] *
    lindlad_hermitian[2][2, -3, -4, -9, -10, 3] *
    lindlad_hermitian[3][3, -5, -6, -11, -12, 4] *
    boundaryR[4];

# @tensor lindbladCheck[-1 -2 -3 -4; -5 -6 -7 -8] := boundaryL[1] * lindblad[1][1, -1, -2, -5, -6, 2] * lindblad[2][2, -3, -4, -7, -8, 3] * boundaryR[3];

lindbladMatrix = reshape(convert(Array, lindbladCheck), (64, 64));
lindbladMatrix /= norm(lindbladMatrix);
open("ising_lindblad.txt", "w") do file
    for row in 1:size(lindbladMatrix, 1)
        for col in 1:size(lindbladMatrix, 2)
            @printf(
                file,
                "(%.3f + %.3fim)",
                real(lindbladMatrix[row, col]),
                imag(lindbladMatrix[row, col])
            )
            if col < size(lindbladMatrix, 2)
                print(file, " ")  # Add space between columns
            else
                println(file)  # New line after each row
            end
        end
    end
end
