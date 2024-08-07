"""
Dynamical simulations of QCP for short chain N = 11
"""

include("../src/lptn.jl")
include("../src/tebd.jl")

using Plots
using ProgressBars
using ColorSchemes
using Serialization
using LaTeXStrings

OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/outputs/"

N = 10;
nTimeSteps = 50;
dt = 0.10; # 0.01 ≤ dt ≤ 0.1
# OMEGAS = LinRange(5, 7, 5);
OMEGAS = [2.0, 6.0, 10.0];
GAMMA = 1.0;

truncErr = 1e-6;
BONDDIM = 50;
KRAUSDIM = 25;

basis0 = [1, 0];
basis1 = [0, 1];
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

for OMEGA in OMEGAS
    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_K_$(KRAUSDIM)"
    n_t = zeros(nTimeSteps + 1)
    ϵHTrunc_t = zeros(nTimeSteps)
    ϵDTrunc_t = zeros(nTimeSteps)
    n_sites_t = Array{Float64}(undef, nTimeSteps + 1, N)

    basisTogether = vcat(fill([basis0, basis1, basis1, basis1, basis1], 2)...)
    # XInit = createXBasis(N, [fill(basis0, N ÷ 2); [Vector{Int64}(basis1)]; fill(basis0, N ÷ 2)]);
    # XInit = createXBasis(N, [fill(basis0, N ÷ 2); fill(basis1, N ÷ 2)]);
    # XInit = createXBasis(N, [basisTogether; fill(basis1, 3)]);
    XInit = createXBasis(N, basisTogether)

    n_sites_t[1, :], n_t[1] = computeSiteExpVal(XInit, numberOp)
    println("Initial average number of particles: $(n_t[1])")

    # run dynamics simulation
    hamDyn = expHam(OMEGA, dt / 2)
    dissDyn = expDiss(GAMMA, dt)
    X_t = XInit
    for i in ProgressBar(1:nTimeSteps)
        # global  X_t
        X_t, ϵHTrunc, ϵDTrunc = TEBD(X_t, hamDyn, dissDyn, BONDDIM, KRAUSDIM, truncErr)
        @show computeNorm(X_t)
        ϵHTrunc_t[i], ϵDTrunc_t[i] = ϵHTrunc, ϵDTrunc
        # println("At t=$(i*dt)")
        n_sites_t[i + 1, :], n_t[i + 1] = computeSiteExpVal(X_t, numberOp)
        # println("Number particles at each site $(n_sites_t[i+1, :])")
    end

    open(OUTPUT_PATH * FILE_INFO * "_n_sites_t.dat", "w") do file
        serialize(file, n_sites_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_n_t.dat", "w") do file
        serialize(file, n_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_H_trunc_err_t.dat", "w") do file
        serialize(file, ϵHTrunc_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_D_trunc_err_t.dat", "w") do file
        serialize(file, ϵDTrunc_t)
    end

    # density plot
    # heatmap(reverse(n_sites_t, dims=1), colorbar=true, colormap=:Greys, yticks=(1:5:nTimeSteps+1, nTimeSteps+1:-5:1),
    #                 xlabel="Site",ylabel="Time",
    #                 title=L"\textrm{Average~number~of~particles~for~} \Omega = %$OMEGA \textrm{~and~} \chi = K = %$BONDDIM")
    # savefig(OUTPUT_PATH * FILE_INFO * "_density_plot.png")
end
