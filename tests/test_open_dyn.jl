"""
Benchmark against dissipative XXZ model
Ref: 10.1103/PhysRevLett.116.237201
"""
include("../src/models.jl")

using TensorKit

BONDDIM = 60
KRAUSDIM = 60
GAMMA = 1.0
DELTA 