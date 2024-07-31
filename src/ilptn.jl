using TensorKit
using KrylovKit # Lanczos - real EVal, Oddrnoldi - complex Eval


function computeSiteExpVal!(Go, Ge, Lo, Le, onSiteOp)
    """
    Args:
    - X : left-canonical MPO    
    """
    
    # for odd site -> bondTensor = Le - Go - Lo
    @tensor bondTensorO[-1 -2;-3 -4] := Le[-1, 1] * Go[1, -2, -3, 2] * Lo[2, -4];
    @tensor expValO = onSiteOp[2, 1] * bondTensorO[3, 4, 2, 5] * conj(bondTensorO[3, 4, 1, 5]);
    expValO /= norm(bondTensorO);

    # for even site -> bondTensor = Lo - Ge - Le
    @tensor bondTensorE[-1 -2;-3 -4] := Lo[-1, 1] * Ge[1, -2, -3, 2] * Le[2, -4];
    @tensor expValE = onSiteOp[2, 1] * bondTensorE[3, 4, 2, 5] * conj(bondTensorE[3, 4, 1, 5]);
    expValE /= norm(bondTensorE);

    if abs(imag(expValO)) < 1e-12 && abs(imag(expValE)) < 1e-12
        return (1/2) * (real(expValO) + real(expValE))
    else
        ErrorException("Oops! Complex expectation value is found.")
    end    
end


function orthogonalizeiMPS!(bondTensor, weightSide, transferOpL, transferOpR)

end