include("../src/models.jl")
include("../src/lptn.jl")
include("../src/tebd.jl")


using TensorKit
using LinearAlgebra
using Plots


N = 3
OMEGA = 6.0;
GAMMA = 1.0;

BONDDIM = 10;
KRAUSDIM = 10;
truncErr = 1e-6;

dt = 0.01;
nTimeSteps = 10;

basis0 = [1, 0];
basis1 = [0, 1];
Id = [1 0; 0 1];
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);


liouvSupOp = constructLiouvMPO(OMEGA, GAMMA, N)
boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
@tensor liouvSupOp[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] :=
    boundaryL[1] *
    liouvSupOp[1][1, -1, -2, -7, -8, 2] *
    liouvSupOp[2][2, -3, -4, -9, -10, 3] *
    liouvSupOp[3][3, -5, -6, -11, -12, 4] *
    boundaryR[4];

liouvMat = reshape(convert(Array, liouvSupOp), (64, 64));

# calculate local observable in terms of density matrix
basisTogether = vcat(fill([basis0, basis1, basis1], N ÷ 3)...)
XInit = createXBasis(N, basisTogether)
@show computeSiteExpVal!(XInit, numberOp) 

# create density matrix from X
boundaryL = TensorMap(ones, ComplexSpace(1), ComplexSpace(1))
boundaryR = TensorMap(ones, ComplexSpace(1), ComplexSpace(1))
@tensor rho[-1 -2 -3; -4 -5 -6] := boundaryL[8, 1] * XInit[1][1, 2, -4, 3] * conj(XInit[1][8, 2, -1, 9]) * 
                                   XInit[2][3, 4, -5, 5] * conj(XInit[2][9, 4, -2, 10]) *
                                   XInit[3][5, 6, -6, 7] * conj(XInit[3][10, 6, -3, 11]) *
                                   boundaryR[7, 11]

rhoVec = reshape(permutedims(convert(Array, rho), (1, 4, 2, 5, 3, 6)), (64, 1))


Id = [1 0; 0 1];
numberOp = [0 0; 0 1];
numberOps = Vector{Array}(undef, N)

for i = 1:N
    numberOpSite = Vector{TensorMap}(undef, N)
    for j = 1:N
        if j == i
            numberOpSite[j] = TensorMap(numberOp, ℂ^1 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^1)
        else
            numberOpSite[j] = TensorMap(Id, ℂ^1 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^1)
        end
    end
    
    boundaryL = TensorMap(ones, one(ComplexSpace()), ComplexSpace(1))
    boundaryR = TensorMap(ones, ComplexSpace(1), one(ComplexSpace()))
    @tensor numberOpSite[-1 -2 -3; -4 -5 -6] := boundaryL[1] * numberOpSite[1][1, -1, -4, 2] *
                                                numberOpSite[2][2, -2, -5, 3] *
                                                numberOpSite[3][3, -3, -6, 4] *
                                                boundaryR[4]
    numberOps[i] = reshape(permutedims(convert(Array, numberOpSite), (1, 4, 2, 5, 3, 6)), (64, 1))
end

siteOcc = Vector{Float64}(undef, N)
for j = 1 : N
    siteOcc[j] = real(dot(numberOps[j], rhoVec))
end
@show siteOcc

avOccED_t = Vector{Float64}(undef, nTimeSteps + 1)
avOccED_t[1] = sum(siteOcc)/3

let
    rhoVec = reshape(permutedims(convert(Array, rho), (1, 4, 2, 5, 3, 6)), (64, 1))
    println("Full time evolution with exact diagonalization")
    for i = 1 : nTimeSteps
        rhoVec = exp(liouvMat * dt) * rhoVec
        siteOcc = Vector{Float64}(undef, N)
        for j = 1 : N
            siteOcc[j] = real(dot(numberOps[j], rhoVec))
        end
        
        avOccED_t[i + 1] = sum(siteOcc) / N
        
    end
end

numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);
avOccTEBD_t = Vector{Float64}(undef, nTimeSteps + 1)
avOccTEBD_t[1] = computeSiteExpVal!(XInit, numberOp)[2]

let
    X_t = XInit
    hamDyn = expCPHam(OMEGA, dt / 2)
    dissDyn = expCPDiss(GAMMA, dt)
    println("Full time evolution with TEBD + LPTN")
    for i = 1 : nTimeSteps
        X_t, ϵHTrunc, ϵDTrunc = TEBD(
                X_t,
                hamDyn,
                dissDyn,
                BONDDIM,
                KRAUSDIM;
                truncErr=truncErr,
                canForm=true, # orthonormalized X_t
            )
        _, avOccTEBD_t[i + 1] = computeSiteExpVal!(X_t, numberOp)
        X_t = orthonormalizeX!(X_t; orthoCenter=1)        
    end
end

aplot = plot();
plot!(dt * (1:(nTimeSteps + 1)), avOccED_t, ls=:dashdot; label="ED")
plot!(dt * (1:(nTimeSteps + 1)), avOccTEBD_t, ls=:dash; label="TEBD+LPTN; CHI=$(BONDDIM), KRAUSDIM=$(KRAUSDIM)")
OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/demos/"
savefig(aplot, OUTPUT_PATH * "test_ED_vs_TEBD_LPTN.pdf")





nothing