"""
For TFI model
Ref: https://tenpy.readthedocs.io/en/latest/toycodes/solution_3_dmrg.html
"""

include("../src/dmrgMPS.jl")

L = 12
CHI = 15
gs = [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
gs = [1.2]

J = 1.0
corrs = []

for g in gs
    tfiMPO = constructTFIMPO(J, g, L)

    # initialize MPS
    initialMPS = initializeRandomMPS(L)

    # run DMRG
    gsMPS, gsEnergy = DMRG2(
        initialMPS, tfiMPO; bondDim=CHI, truncErr=1e-10, convTolE=1e-10, verbosePrint=true
    )
    println("ground state energy per site E = $(gsEnergy/L)")

    Sz = [+1 0; 0 -1]
    Sz = TensorMap(Sz, ComplexSpace(2), ComplexSpace(2))
    @show computeExpVal1(gsMPS, Sz) # magnetization in Z = 0.00000

    Sx = [0 +1; +1 0]
    Sx = TensorMap(Sx, ComplexSpace(2), ComplexSpace(2))
    @show computeExpVal1(gsMPS, Sx) # magnetization in X = 0.81887

    @show computeCorr2!(gsMPS, Sz, 3, 9) # 0.08762
end
