using Plots
using Printf
using ColorSchemes
using Serialization
using LaTeXStrings

using TensorKit
using NPZ

include("../src/lptn.jl")
default(; fontfamily="Computer Modern")
colorPal = palette(:tab10)

N = 5;
nTimeSteps = 200;
dt = 0.1; # 0.01 ≤ dt ≤ 0.1
JOBID = 1004846;
CHI = 30;
KRAUSDIM = 30;
CHIS = [30, 50, 70];
KRAUSDIMS = [30, 50, 70];
# CHIS = [100, 200, 300]; KRAUSDIMS = [50, 100, 150]

OMEGAS = [0.0, 0.9, 1.9, 2.8, 3.8, 4.7, 5.7, 6.7, 7.6, 8.6, 9.5, 10.5];
# OMEGAS = [0.0, 0.9, 1.9, 3.8, 4.7, 5.7, 6.7, 7.6, 8.6, 9.5, 10.5];

GAMMA = 1.0;

RESULT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/results/"
OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/outputs/"
FILES = "N_$(N)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";
FILE = "N_$(N)_dt_$(dt)_ntime_$(nTimeSteps)_$(JOBID)";

## load files
ϵHTrunc_t = Any[];
ϵDTrunc_t = Any[];
dens_corr_s = Any[];

for (i, OMEGA) in enumerate(OMEGAS)
    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)"

    push!(ϵHTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_H_trunc_err_t.dat"))
    push!(ϵDTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_D_trunc_err_t.dat"))
    push!(dens_corr_s, deserialize(RESULT_PATH * FILE_INFO * "_dens_dens_corr.dat"))
end

n_t_s = Array{Float64}(undef, length(CHIS), length(OMEGAS), nTimeSteps + 1);

ent_entropy_t_s = Array{Float64}(undef, length(CHIS), length(OMEGAS), nTimeSteps + 1);
renyi_entropy_t_s = Array{Float64}(undef, length(CHIS), length(OMEGAS), nTimeSteps + 1);
puri_ent_t_s = Array{Float64}(undef, length(CHIS), length(OMEGAS), nTimeSteps + 1);

n_qs_s = Array{Float64}(undef, length(CHIS), length(OMEGAS));

for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHIS[i])_K_$(KRAUSDIMS[i])_$(JOBID)"
        n_t_s[i, j, :] = deserialize(RESULT_PATH * FILE_INFO * "_n_t.dat")
        n_qs_s[i, j] = n_t_s[i, j, end]

        ent_entropy_t_s[i, j, :] = deserialize(
            RESULT_PATH * FILE_INFO * "_ent_entropy_t.dat"
        )
        renyi_entropy_t_s[i, j, :] = deserialize(RESULT_PATH * FILE_INFO * "_renyiMI_t.dat")
        puri_ent_t_s[i, j, :] = deserialize(RESULT_PATH * FILE_INFO * "_puriEnt_t.dat")
    end
end

# # n_qs
aplot = plot();
ytick_positions = [1.0, 0.1, 0.01]
ytick_labels = ["1.0", "0.1", "0.01"]

for i in eachindex(CHIS)
    plot!(OMEGAS, n_qs_s[i, :]; label=L"(\chi, \, K) = (%$(CHIS[i]), %$(KRAUSDIMS[i]))")
end

plot!(;
    xlabel=L"\Omega",
    ylabel=L"\textrm{n_{qs}}",
    title=L"N=%$N",
    legend=:topright,
    yformatter=:scientific,
)
savefig(aplot, OUTPUT_PATH * FILE * "_n_qs.pdf")

# n(t) for OMEGA = 1.9, 5.7, 10.5
aplot = plot();
ytick_positions = [1.0, 0.1, 0.01]
ytick_labels = ["1.0", "0.1", "0.01"]
for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        if OMEGA in [1.9, 5.7, 10.5]
            plot!(
                dt * (1:(nTimeSteps + 1)),
                n_t_s[i, j, :];
                label=L"(\chi, \, K) = (%$(CHIS[i]), %$(KRAUSDIMS[i]))",
                ls=(
                    if i == 1
                        :dashdot
                    elseif i == 2
                        :dash
                    else
                        :solid
                    end
                ),
                lc=(
                    if OMEGA == 1.9
                        colorPal[1] # blue
                    elseif OMEGA == 5.7
                        colorPal[2] # orange
                    else
                        colorPal[3] # green
                    end
                ),
            )
        end
    end
end
plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{n(t)}",
    title=L"N=%$N",
    legend=:bottomleft,
    xaxis=:log10,
    yaxis=:log10,
    yticks=(ytick_positions, ytick_labels),
    xformatter=(val) -> @sprintf("%.1f", val),
    yformatter=(val) -> @sprintf("%.2f", val),
)
savefig(aplot, OUTPUT_PATH * FILE * "_sub_crit_super_n(t).pdf")

# av(t)
aplot = plot();
ytick_positions = [1.0, 0.1, 0.01]
ytick_labels = ["1.0", "0.1", "0.01"]
index_CHI = findfirst(==(CHI), CHIS)
for (j, OMEGA) in enumerate(OMEGAS)
    plot!(dt * (1:(nTimeSteps + 1)), n_t_s[index_CHI, j, :]; label=L"\Omega = %$OMEGA")
end
plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{n(t)}",
    title=L"\chi=%$CHI, \, K=%$KRAUSDIM, \, N=%$N",
    legend=:bottomleft,
    xaxis=:log10,
    yaxis=:log10,
    yticks=(ytick_positions, ytick_labels),
    xformatter=(val) -> @sprintf("%.1f", val),
    yformatter=(val) -> @sprintf("%.2f", val),
)
savefig(aplot, OUTPUT_PATH * FILES * "_av_n(t).pdf")

## density-density correlation
aplot = plot();

for (i, OMEGA) in enumerate(OMEGAS)
    plot!((2:N), dens_corr_s[i]; label=L"\Omega = %$OMEGA")
end

plot!(;
    xlabel=L"r",
    ylabel=L"\textrm{C(r)}",
    title=L"N=%$N, \, \chi=%$CHI, \, K=%$KRAUSDIM",
    legend=:topright,
    # xaxis=:log10,
    # yaxis=:log10,
    yformatter=:scientific,
)
savefig(aplot, OUTPUT_PATH * FILES * "_dens_dens_corr.pdf")

# ϵHTrunc
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_sum = [sum(x) for x in ϵHTrunc_t[i]]
    ϵHTrunc_t_max = [maximum(x) for x in ϵHTrunc_t[i]]
    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵHTrunc_t_sum;
        label=L"\Omega = %$OMEGA, \, \textrm{sum. err.}",
    )
    # plot!(
    #     aplot,
    #     dt * (1:nTimeSteps),
    #     ϵHTrunc_t_max;
    #     label=L"\Omega = %$OMEGA, \, \textrm{max. err.}",
    # )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Truncation~error~in~} \chi, \, \chi=%$CHI",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_t.pdf")

## cumulative ϵHTrunc accum.
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_sum = [sum(x) for x in ϵHTrunc_t[i]]
    ϵHTrunc_t_cumsum = cumsum(ϵHTrunc_t_sum) / (nTimeSteps * dt)

    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵHTrunc_t_cumsum;
        label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}",
    )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Cumulative~truncation~error~in~} \chi, \, \chi=%$CHI",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_cumsum_sum_t.pdf")

## cumulative ϵHTrunc max
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_max = [maximum(x) for x in ϵHTrunc_t[i]]
    ϵHTrunc_t_cumsum_max = cumsum(ϵHTrunc_t_max) / (nTimeSteps * dt)

    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵHTrunc_t_cumsum_max;
        label=L"\Omega = %$OMEGA, \, \textrm{max. err.}",
    )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Cumulative~truncation~error~in~} \chi, \, \chi=%$CHI",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_cumsum_sum_max_t.pdf")

# ϵDTrunc
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_sum = [sum(x) for x in ϵDTrunc_t[i]]
    ϵDTrunc_t_max = [maximum(x) for x in ϵDTrunc_t[i]]
    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵDTrunc_t_sum;
        label=L"\Omega = %$OMEGA, \, \textrm{sum. err.}",
    )
    # plot!(
    #     aplot,
    #     dt * (1:nTimeSteps),
    #     ϵDTrunc_t_max;
    #     label=L"\Omega = %$OMEGA, \, \textrm{max. err.}",
    # )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Truncation~error~in~} K, \, K=%$KRAUSDIM",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_t.pdf")

## cumulative ϵDTrunc accum.
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_sum = [sum(x) for x in ϵDTrunc_t[i]]
    ϵDTrunc_t_cumsum = cumsum(ϵDTrunc_t_sum) / (nTimeSteps * dt)

    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵDTrunc_t_cumsum;
        label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}",
    )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Cumulative~truncation~error~in~} K, \, K=%$KRAUSDIM",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_cumsum_sum_t.pdf")

## cumulative ϵDTrunc max
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_max = [maximum(x) for x in ϵDTrunc_t[i]]
    ϵDTrunc_t_cumsum_max = cumsum(ϵDTrunc_t_max) / (nTimeSteps * dt)

    plot!(
        aplot,
        dt * (1:nTimeSteps),
        ϵDTrunc_t_cumsum_max;
        label=L"\Omega = %$OMEGA, \, \textrm{max. err.}",
    )
end

plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Cumulative~truncation~error~in~} K, \, K=%$KRAUSDIM",
    title=L"N=%$N",
)
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_cumsum_sum_max_t.pdf")

# entanglement entropy
aplot = plot();
for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        if OMEGA in [1.9, 5.7, 10.5]
            plot!(
                dt * (1:(nTimeSteps + 1)),
                ent_entropy_t_s[i, j, :];
                label=L"(\chi, \, K) = (%$(CHIS[i]), %$(KRAUSDIMS[i]))",
                ls=(
                    if i == 1
                        :dashdot
                    elseif i == 2
                        :dash
                    else
                        :solid
                    end
                ),
                lc=(
                    if OMEGA == 1.9
                        colorPal[1]
                    elseif OMEGA == 5.7
                        colorPal[2]
                    else
                        colorPal[3]
                    end
                ),
            )
        end
    end
end
plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Bipartite~entanglement~(von~Neumann)~entropy}",
    title=L"N=%$N",
    legend=:topright,
)
savefig(aplot, OUTPUT_PATH * FILE * "_vnEnt_t.pdf")

## 2-Renyi entropy
aplot = plot();
for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        if OMEGA in [1.9, 5.7, 10.5]
            plot!(
                dt * (1:(nTimeSteps + 1)),
                renyi_entropy_t_s[i, j, :];
                label=L"(\chi, \, K) = (%$(CHIS[i]), %$(KRAUSDIMS[i]))",
                ls=(
                    if i == 1
                        :dashdot
                    elseif i == 2
                        :dash
                    else
                        :solid
                    end
                ),
                lc=(
                    if OMEGA == 1.9
                        colorPal[1]
                    elseif OMEGA == 5.7
                        colorPal[2]
                    else
                        colorPal[3]
                    end
                ),
            )
        end
    end
end
plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Bipartite~2-Rényi~Mutual~Information}",
    title=L"N=%$N",
    legend=:topright,
)
savefig(aplot, OUTPUT_PATH * FILE * "_renyiMI_t.pdf")

# entanglement of purfication
aplot = plot();
for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        if OMEGA in [1.9, 5.7, 10.5]
            plot!(
                dt * (1:(nTimeSteps + 1)),
                puri_ent_t_s[i, j, :];
                label=L"(\chi, \, K) = (%$(CHIS[i]), %$(KRAUSDIMS[i]))",
                ls=(
                    if i == 1
                        :dashdot
                    elseif i == 2
                        :dash
                    else
                        :solid
                    end
                ),
                lc=(
                    if OMEGA == 1.9
                        colorPal[1]
                    elseif OMEGA == 5.7
                        colorPal[2]
                    else
                        colorPal[3]
                    end
                ),
            )
        end
    end
end
plot!(;
    xlabel=L"\textrm{t}",
    ylabel=L"\textrm{Entanglement~of~purfication}",
    title=L"N=%$N",
    legend=:topright,
    legendfontsize=5,
)
savefig(aplot, OUTPUT_PATH * FILE * "_puriEnt_t.pdf")

### Density plot of the site-resolved average density 
### FIG S2
# OMEGA = 6.0;

# FILE_INFO = "diff_N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";
# FILES = "diff_N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";

# n_sites_t = deserialize(RESULT_PATH * FILE_INFO * "_n_sites_t.dat");

# aplot = plot();
# heatmap(reverse(n_sites_t, dims=1), colorbar=true, colormap=:Greys, yticks=(1:5:nTimeSteps+1, nTimeSteps+1:-5:1),
#                     xlabel="Site",ylabel="Time",
#                     title=L"\textrm{Average~number~of~particles~for~} \Omega = %$OMEGA,\, \chi = %$CHI,\, K=%$KRAUSDIM")
# savefig(OUTPUT_PATH * FILES * "_density_plot.png")

# plot final state
# finalStates = Array{Any}(undef, length(CHIS), 3);
# PATH_TO_SCIKIT_TT = "/home/psireal42/study/qcp-1d-julia/for_scikittt/"
# for i in eachindex(CHIS)
#     global index = 1

#     for (j, OMEGA) in enumerate(OMEGAS)
#         if OMEGA in [1.9, 5.7, 10.5]
#             FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHIS[i])_K_$(KRAUSDIMS[i])_$(JOBID)_finalState.dat"
#             finalStateLoad = deserialize(RESULT_PATH * FILE_INFO)
#             finalState = Vector{TensorMap}(undef, N)

#             for k = 1 : N
#                 finalState[k] = convert(TensorMap, finalStateLoad[k])
#                 finalState_k_Arr = convert(Array, finalState[k])
#                 @show OMEGA
#                 @show CHIS[i], KRAUSDIMS[i]
#                 @show k
#                 @show size(finalState_k_Arr)
#                 # serialize(PATH_TO_SCIKIT_TT * "N_$(N)_OMEGA_$(OMEGA)_CHI_$(CHIS[i])_K_$(KRAUSDIMS[i])_index_$k.dat", finalState_k_Arr)
#                 npzwrite(PATH_TO_SCIKIT_TT * "N_$(N)_OMEGA_$(OMEGA)_CHI_$(CHIS[i])_K_$(KRAUSDIMS[i])_index_$k.npy", finalState_k_Arr)
#             end

#             finalStates[i, index] = finalState
#             index += 1
#         end
#     end
# end

# finalMatrices_CHI_1 = Array{Any}(undef, 3)

# for i = 1 : 3
#     boundaryL = TensorMap(ones, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
#     boundaryL = updateEnvL(1, N, finalStates[1, i], boundaryL)
#     @tensor boundaryL[-1; -2] := boundaryL[-1, 1, -2, 1]
#     boundaryL = convert(Array, boundaryL)

#     finalMatrices_CHI_1[i] = abs.(boundaryL)
# end

# aplot = plot();
# OMEGAS = [1.9, 5.7, 10.5]

# for i = 1:3
#     local FILES = "N_$(N)_OMEGA_$(OMEGAS[i])_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";

#     heatmap(finalMatrices_CHI_1[i], colorbar=true, colormap=:deep,
#                         title=L"\textrm{Density~matrix~at~t=20~for~} \Omega = %$(OMEGAS[i]),\, \chi = %$CHI,\, K=%$KRAUSDIM")
#     savefig(OUTPUT_PATH * FILES * "_finalState.pdf")
# end

# basis0 = [1, 0];
# basisTogether = vcat(fill([basis0, basis0, basis0, basis0, basis0], N ÷ 5)...)
# darkState = createXBasis(N, basisTogether)
# boundaryL = TensorMap(ones, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
# boundaryL = updateEnvL(1, N, darkState, boundaryL)
# @tensor boundaryL[-1; -2] := boundaryL[-1, 1, -2, 1]
# boundaryL = convert(Array, boundaryL)
# darkState = abs.(boundaryL)

# heatmap(darkState, colorbar=true, colormap=:deep,
#                     title="Dark state density matrix")
# savefig(OUTPUT_PATH * "darkState.pdf")
