include("../src/lptn.jl")
include("../src/tebd.jl")
include("../src/models.jl")

using LinearAlgebra
using TensorKit
using Plots
using ColorSchemes
using LaTeXStrings

include("../src/lptn.jl")
default(; fontfamily="Computer Modern")
colorPal = palette(:tab10)

OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/demos/"


N = 12
CHI = 15
KRAUSDIM = 15
basis0 = [1, 0];
basis1 = [0, 1];
delta = 0.01;
nTimeSteps = 1000;
GAMMAS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
FILE = "N_$(N)_delta_$(delta)_ntime_$(nTimeSteps)";

J = 1.0
g = 1.2;

Sx = TensorMap([0 1; 1 0], ℂ^2, ℂ^2);
Sz = TensorMap([1 0; 0 -1], ℂ^2, ℂ^2);
Id = TensorMap([1 0; 0 1], ℂ^2, ℂ^2);
HBulk = -J * (Sz ⊗ Sz) - g / 2 * (Sx ⊗ Id + Id ⊗ Sx);
HL = -J * (Sz ⊗ Sz) - g * Sx ⊗ Id - g / 2 * Id ⊗ Sx;
HR = -J * (Sz ⊗ Sz) - g / 2 * Sx ⊗ Id - g * Id ⊗ Sx;

expHo = exp(-1 * delta / 2 * HBulk);
expHe = exp(-1 * delta * HBulk);
expHL = exp(-1 *delta / 2 * HL)
if N % 2 == 0
    expHR = exp(-1 * delta / 2 * HR)
else
    expHR = exp(-1 * delta * HR)
end
tfiMPO = constructTFIMPO(J, g, N)


### no noise
vNEntropy_t = []
Sx_av_t = []
energy_t = []

let
    basisTogether = vcat(fill([basis0], N)...)
    X_t = createXBasis(N, basisTogether)
    X_t = orthonormalizeX!(X_t; orthoCenter=1)
    _, Sx_av = computeSiteExpVal!(X_t, Sx)
    push!(Sx_av_t, Sx_av)
    # X_t = orthonormalizeX!(X_t; orthoCenter=1)
    # push!(vNEntropy_t, computevNEntropy!(X_t))
    # X_t = orthonormalizeX!(X_t; orthoCenter=1)
    # push!(energy_t, computeEnergy!(X_t, tfiMPO))
    # X_t = orthonormalizeX!(X_t; orthoCenter=1)
       

    for i in 1:nTimeSteps
        X_t, ϵHTrunc, ϵDTrunc = TEBD_noDiss!(X_t, expHL, expHo, expHe, expHR, CHI)
        _, Sx_av = computeSiteExpVal!(X_t, Sx)
        push!(Sx_av_t, Sx_av)
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(vNEntropy_t, computevNEntropy!(X_t))
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(energy_t, computeEnergy!(X_t, tfiMPO))
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
    
    end

    # @show computeSiteExpVal!(X_t, Sx) # magnetization in X = 0.81887
    # @show energy_t[end]
end

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), Sx_av_t)
# plot!(; xlabel=L"t", ylabel=L"\langle \sigma^x \rangle")
# savefig(aplot, OUTPUT_PATH * FILE * "_Sx_no_noise.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), vNEntropy_t)
# plot!(; xlabel=L"t", ylabel=L"S_{vN}")
# savefig(aplot, OUTPUT_PATH * FILE * "_SvN_no_noise.pdf")

# ### with dephasing noise
# vNEntropy_dep_t = []
# Sx_av_dep_t = []

# function expDephasing(gamma, delta)
#     Sz = [+1 0; 0 -1]
#     Id = [+1 0; 0 +1]

#     diss =
#         gamma * (kron(Sz, Sz) - kron(Id, Id))
#     diss = TensorMap(
#         diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
#     )
#     diss = exp(delta * diss)

#     # EVD    
#     D, V = eig(diss, (1, 3), (2, 4))
#     B = permute(sqrt(D) * V', (3, 1), (2,))

#     return B
# end

expHo = exp(-1 * delta / 2 * HBulk);
expHe = exp(-1 * delta / 2 * HBulk);
expHL = exp(-1 * delta / 2 * HL)
expHR = exp(-1 * delta / 2 * HR)
# noise = expDephasing(0.5, delta)

# let
#     basisTogether = vcat(fill([basis0], N)...)
#     X_t = createXBasis(N, basisTogether)
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)
#     _, Sx_av = computeSiteExpVal!(X_t, Sx)
#     push!(Sx_av_dep_t, Sx_av)
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)
#     push!(vNEntropy_dep_t, computevNEntropy!(X_t))
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)    
    
#     for i in 1:nTimeSteps
#         X_t, ϵHTrunc, ϵDTrunc = TEBD_OBC_Diss!(X_t, expHL, expHo, expHe, expHR, noise, 15, 15)
#         _, Sx_av = computeSiteExpVal!(X_t, Sx)
#         push!(Sx_av_dep_t, Sx_av)
#         X_t = orthonormalizeX!(X_t; orthoCenter=1)
#         push!(vNEntropy_dep_t, computevNEntropy!(X_t))
#         X_t = orthonormalizeX!(X_t; orthoCenter=1)    
#     end
# end


# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), Sx_av_dep_t)
# plot!(; xlabel=L"t", ylabel=L"\langle \sigma^x \rangle")
# savefig(aplot, OUTPUT_PATH * FILE * "_Sx_dephasing_noise.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), vNEntropy_dep_t)
# plot!(; xlabel=L"t", ylabel=L"S_{vN}")
# savefig(aplot, OUTPUT_PATH * FILE * "_SvN_dephasing_noise.pdf")


# ### with spontaneous emission
# function expSponEm(gamma, delta)
#     """
#     Dissipative dynamics at the rightmost site - drain
#     L = √(2 * γ) * σ^-
#     """
#     Sx = [0 +1; +1 0]
#     Sy = [0 -1im; +1im 0]
#     Splus = (Sx + 1im * Sy) / 2 # [0 1; 0 0]
#     Sminus = (Sx - 1im * Sy) / 2 # [0 0; 1 0]
#     Id = [+1 0; 0 +1]

#     diss =
#         gamma * (
#             kron(Sminus, Sminus) - (1 / 2) * kron((Splus * Sminus), Id) -
#             (1 / 2) * kron(Id, (Splus * Sminus))
#         )
#     diss = TensorMap(
#         diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
#     )
#     diss = exp(delta * diss)

#     # EVD    
#     D, V = eig(diss, (1, 3), (2, 4))
#     B = permute(sqrt(D) * V', (3, 1), (2,))

#     return B
# end

# noise = expSponEm(1.5, delta)
# vNEntropy_sponEm_t = []
# Sx_av_sponEm_t = []

# let
#     basisTogether = vcat(fill([basis0], N)...)
#     X_t = createXBasis(N, basisTogether)
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)
#     _, Sx_av = computeSiteExpVal!(X_t, Sx)
#     push!(Sx_av_sponEm_t, Sx_av)
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)
#     push!(vNEntropy_sponEm_t, computevNEntropy!(X_t))
#     X_t = orthonormalizeX!(X_t; orthoCenter=1)    
    
#     for i in 1:nTimeSteps
#         X_t, ϵHTrunc, ϵDTrunc = TEBD_OBC_Diss!(X_t, expHL, expHo, expHe, expHR, noise, 15, 15)
#         _, Sx_av = computeSiteExpVal!(X_t, Sx)
#         push!(Sx_av_sponEm_t, Sx_av)
#         X_t = orthonormalizeX!(X_t; orthoCenter=1)
#         push!(vNEntropy_sponEm_t, computevNEntropy!(X_t))
#         X_t = orthonormalizeX!(X_t; orthoCenter=1)    
#     end
# end

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), Sx_av_sponEm_t)
# plot!(; xlabel=L"t", ylabel=L"\langle \sigma^x \rangle")
# savefig(aplot, OUTPUT_PATH * FILE * "_Sx_sponEm_noise.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), vNEntropy_sponEm_t)
# plot!(; xlabel=L"t", ylabel=L"S_{vN}")
# savefig(aplot, OUTPUT_PATH * FILE * "_SvN_sponEm_noise.pdf")

### with random unitary noise
function randUnifUnit(dim::Int)
    A = rand(ComplexF64, dim, dim)
    Q, R = qr(A)
    Q *= Diagonal(sign.(diag(R)))

    return Q
end

function randomUniNoise(gamma, delta)
    """
    Ref: DOI: 10.1103/PhysRevResearch.3.023005 [Eq. 2.5]

    Params:
    - 0 <= p <= 1
    """
    randMat = randUnifUnit(2)
    Id = [+1 0; 0 +1]

    noise = gamma * (
        kron(randMat, conj(randMat)) - (1 / 2) * kron(Id, Id) -
        (1 / 2) * kron(Id, (transpose(randMat) * conj(randMat)))
    ) 
    # + (1 - gamma)  * kron(Id, Id)
    noise = TensorMap(
        noise, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
    )

    noise = exp(delta * noise)

    # EVD    
    D, V = eig(noise, (1, 3), (2, 4))
    B = permute(sqrt(D) * V', (3, 1), (2,))

    return B
end

vNEntropy_randomUni_t_many = []
Sx_av_randomUni_t_many = []
energy_randomUni_t_many = []

for gamma in GAMMAS
    noise = randomUniNoise(gamma, -delta)
    vNEntropy_randomUni_t = []
    Sx_av_randomUni_t = []
    energy_randomUni_t = []

    let
        basisTogether = vcat(fill([basis0], N)...)
        X_t = createXBasis(N, basisTogether)
        X_t = orthonormalizeX!(X_t; orthoCenter=1)
        _, Sx_av = computeSiteExpVal!(X_t, Sx)
        push!(Sx_av_randomUni_t, Sx_av)
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(vNEntropy_randomUni_t, computevNEntropy!(X_t))
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(energy_randomUni_t, computeEnergy!(X_t, tfiMPO))
        X_t = orthonormalizeX!(X_t; orthoCenter=1)      
        
        for i in 1:nTimeSteps
            X_t, ϵHTrunc, ϵDTrunc = TEBD_OBC_Diss!(X_t, expHL, expHo, expHe, expHR, noise, CHI, KRAUSDIM)
            _, Sx_av = computeSiteExpVal!(X_t, Sx)
            push!(Sx_av_randomUni_t, Sx_av)
            X_t = orthonormalizeX!(X_t; orthoCenter=1)
            # push!(vNEntropy_randomUni_t, computevNEntropy!(X_t))
            # X_t = orthonormalizeX!(X_t; orthoCenter=1)
            # push!(energy_randomUni_t, computeEnergy!(X_t, tfiMPO))
            # X_t = orthonormalizeX!(X_t; orthoCenter=1)    
        end
        
        # push!(vNEntropy_randomUni_t_many, vNEntropy_randomUni_t)
        push!(Sx_av_randomUni_t_many, Sx_av_randomUni_t)
        # push!(energy_randomUni_t_many, energy_randomUni_t)

    end
end

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), Sx_av_randomUni_t)
# plot!(; xlabel=L"t", ylabel=L"\langle \sigma^x \rangle", title=L"(\chi, \, K) = (%$(CHI), %$(KRAUSDIM))")
# savefig(aplot, OUTPUT_PATH * FILE * "_Sx_randomUnitary_noise.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), vNEntropy_randomUni_t)
# plot!(; xlabel=L"t", ylabel=L"S_{vN}")
# savefig(aplot, OUTPUT_PATH * FILE * "_SvN_randomUnitary_noise.pdf")

### with random non-unitary noise
function randomNoise(gamma, delta)
    """
    Params:
    - 0 <= gamma <= 1
    """
    randMat = randuniform(ComplexF64, (2,2))
    # randMat = randnormal(ComplexF64, (2,2))

    Id = [+1 0; 0 +1]

    noise = gamma * (
        kron(randMat, conj(randMat)) - (1 / 2) * kron(randMat'*randMat, Id) -
        (1 / 2) * kron(Id, (transpose(randMat) * conj(randMat)))
    ) 
    # + (1 - gamma)  * kron(Id, Id)
    noise = TensorMap(
        noise, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)'
    )

    noise = exp(delta * noise)

    # EVD    
    D, V = eig(noise, (1, 3), (2, 4))
    B = permute(sqrt(D) * V', (3, 1), (2,))

    return B
end

vNEntropy_randomNoise_t_many = []
Sx_av_randomNoise_t_many = []
energy_randomNoise_t_many = []


for gamma in GAMMAS
    noise = randomNoise(gamma, -delta)
    vNEntropy_randomNoise_t = []
    Sx_av_randomNoise_t = []
    energy_randomNoise_t = []

    let
        basisTogether = vcat(fill([basis0], N)...)
        X_t = createXBasis(N, basisTogether)
        X_t = orthonormalizeX!(X_t; orthoCenter=1)
        _, Sx_av = computeSiteExpVal!(X_t, Sx)
        push!(Sx_av_randomNoise_t, Sx_av)
        X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(vNEntropy_randomNoise_t, computevNEntropy!(X_t))
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)
        # push!(energy_randomNoise_t, computeEnergy!(X_t, tfiMPO))
        # X_t = orthonormalizeX!(X_t; orthoCenter=1)       
        
        for i in 1:nTimeSteps
            X_t, ϵHTrunc, ϵDTrunc = TEBD_OBC_Diss!(X_t, expHL, expHo, expHe, expHR, noise, CHI, KRAUSDIM)
            _, Sx_av = computeSiteExpVal!(X_t, Sx)
            push!(Sx_av_randomNoise_t, Sx_av)
            X_t = orthonormalizeX!(X_t; orthoCenter=1)
            # push!(vNEntropy_randomNoise_t, computevNEntropy!(X_t))
            # X_t = orthonormalizeX!(X_t; orthoCenter=1)
            # push!(energy_randomNoise_t, computeEnergy!(X_t, tfiMPO))
            # X_t = orthonormalizeX!(X_t; orthoCenter=1)      
        end
    end

    # push!(vNEntropy_randomNoise_t_many, vNEntropy_randomNoise_t)
    push!(Sx_av_randomNoise_t_many, Sx_av_randomNoise_t)
    # push!(energy_randomNoise_t_many, energy_randomNoise_t)
end

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), Sx_av_randomNoise_t_many[1], ls=:dashdot, label="uniform")
# plot!(delta * (0:(nTimeSteps)), Sx_av_randomUni_t_many[1], ls=:dash, label="uniform+unitary")
# plot!(delta * (0:(nTimeSteps)), Sx_av_t, label="no noise")
# plot!(; xlabel=L"t", ylabel=L"\langle \sigma^z \rangle",
#         title=L"(N, \chi, \, K) = (%$(N),%$(CHI), %$(KRAUSDIM))")
# savefig(aplot, OUTPUT_PATH * FILE * "_for_checking_imag.pdf")


aplot = plot();
plot!(delta * (0:(nTimeSteps)), Sx_av_t, label="no noise")
for i = 2:6
    plot!(delta * (0:(nTimeSteps)), Sx_av_randomNoise_t_many[i], ls=:dashdot, label=L"uniform, \gamma=%$(GAMMAS[i])")
    plot!(delta * (0:(nTimeSteps)), Sx_av_randomUni_t_many[i], ls=:dash, label=L"uniform+unitary, \gamma=%$(GAMMAS[i])")
end
plot!(; xlabel=L"t", ylabel=L"\langle \sigma^z \rangle", legend=:bottomright, legendfontsize=7,
        title=L"(N, \chi, \, K) = (%$(N),%$(CHI), %$(KRAUSDIM))")

savefig(aplot, OUTPUT_PATH * FILE * "_Sx_zusammen_imag.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), vNEntropy_t, label="no noise")
# for i = 2:6
#     plot!(delta * (0:(nTimeSteps)), vNEntropy_randomNoise_t_many[i], ls=:dashdot, label=L"uniform, \gamma=%$(GAMMAS[i])")
#     plot!(delta * (0:(nTimeSteps)), vNEntropy_randomUni_t_many[i], ls=:dash, label=L"uniform+unitary, \gamma=%$(GAMMAS[i])")
# end
# plot!(; xlabel=L"t", ylabel=L"S_{vN}", legend=:bottomright, legendfontsize=7,
#         title=L"(N, \chi, \, K) = (%$(N),%$(CHI), %$(KRAUSDIM))")
# savefig(aplot, OUTPUT_PATH * FILE * "_SvN_zusammen_imag.pdf")

# aplot = plot();
# plot!(delta * (0:(nTimeSteps)), energy_t, label="no noise")
# for i = 2:6
#     plot!(delta * (0:(nTimeSteps)), energy_randomNoise_t_many[i], ls=:dashdot, label=L"uniform, \gamma=%$(GAMMAS[i])")
#     plot!(delta * (0:(nTimeSteps)), energy_randomUni_t_many[i], ls=:dash, label=L"uniform+unitary, \gamma=%$(GAMMAS[i])")
# end
# plot!(; xlabel=L"t", ylabel=L"Energy", legend=:topright, legendfontsize=7,
#         title=L"(N, \chi, \, K) = (%$(N),%$(CHI), %$(KRAUSDIM))")
# savefig(aplot, OUTPUT_PATH * FILE * "_energy_zusammen_imag.pdf")