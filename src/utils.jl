"""
Utility functions for LPTN Ansatz
"""

function tensorSize(tensorNW::Vector{TensorMap})
    """
    Return size of tensor network in bytes
    """
    size = 0;
    for (i, tensor) in enumerate(tensorNW)
        size += Base.summarysize(tensor);
    end

    return size
end