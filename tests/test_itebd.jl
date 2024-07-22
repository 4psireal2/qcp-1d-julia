using LinearAlgebra
using TensorKit

include("../src/itebd.jl")


d = 2; 
bondDim = 2; 

oddT = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
evenT = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));

# weight matrices
wOddEven = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));
wOddEven /= norm(wOddEven);
wEvenOdd = TensorMap(ones, ComplexSpace(bondDim), ComplexSpace(bondDim));
wEvenOdd /= norm(wEvenOdd);

# transfer operators
leftT_EO_init = Matrix{ComplexF64}(I, bondDim, bondDim) ./ bondDim;
leftT_EO_init = TensorMap(leftT_EO_init, ComplexSpace(bondDim), ComplexSpace(bondDim));
rightT_OE_init = Matrix{ComplexF64}(I, bondDim, bondDim) ./ bondDim;
rightT_OE_init = TensorMap(rightT_OE_init, ComplexSpace(bondDim), ComplexSpace(bondDim));

_, leftT_EO = leftContraction!(oddT, evenT, wOddEven, wEvenOdd, leftT_EO_init);
rightT_OE, _ = rightContraction!(oddT, evenT, wOddEven, wEvenOdd, rightT_OE_init);

leftT_EO_init = Matrix{ComplexF64}(I, bondDim, bondDim) ./ bondDim;
leftT_EO_init = TensorMap(leftT_EO_init, ComplexSpace(bondDim), ComplexSpace(bondDim));
rightT_OE_init = Matrix{ComplexF64}(I, bondDim, bondDim) ./ bondDim;
rightT_OE_init = TensorMap(rightT_OE_init, ComplexSpace(bondDim), ComplexSpace(bondDim));

_, leftT_EO_test = leftContraction_test!(oddT, evenT, wOddEven, wEvenOdd, leftT_EO);
rightT_OE_test, _ = rightContraction_test!(oddT, evenT, wOddEven, wEvenOdd, rightT_OE);

leftT_EO_matrix = reshape(convert(Array, leftT_EO), bondDim, bondDim);
leftT_EO_test_matrix = reshape(convert(Array, leftT_EO_test), bondDim, bondDim);

rightT_OE_matrix = reshape(convert(Array, rightT_OE), bondDim, bondDim);
rightT_OE_test_matrix = reshape(convert(Array, rightT_OE_test), bondDim, bondDim);

@show leftT_EO_matrix
@show leftT_EO_test_matrix
@assert isapprox(leftT_EO_matrix, leftT_EO_test_matrix);

@show rightT_OE_matrix
@show rightT_OE_test_matrix
@assert isapprox(rightT_OE_matrix, rightT_OE_test_matrix);

@show orthogonalizeiMPS!(oddT, evenT, wOddEven, leftT_EO, rightT_OE)

nothing
