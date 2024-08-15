using Plots
using Printf
using ColorSchemes
using Serialization
using LaTeXStrings

include("lptn.jl")

default(; fontfamily="Computer Modern")
colorPal = palette(:tab10)

N = 10;
nTimeSteps = 200;
dt = 0.1; # 0.01 ≤ dt ≤ 0.1
JOBID = 738838;
CHI = 200;
KRAUSDIM = 100;
CHIS = [100, 200]
KRAUSDIMS = [50, 100]

OMEGAS = [0.0, 0.9, 1.9, 2.8, 3.8, 4.7, 5.7, 6.7, 7.6, 8.6, 9.5, 10.5];
GAMMA = 1.0;

RESULT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/results/"
OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/outputs/"
FILES = "N_$(N)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";

## load files
ϵHTrunc_t = Any[];
ϵDTrunc_t = Any[];

for (i, OMEGA) in enumerate(OMEGAS)
    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)"

    # n_t_s[i, :] = deserialize(RESULT_PATH * FILE_INFO * "_n_t.dat")
    # n_qs_s[i] = n_t_s[i, end]

    push!(ϵHTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_H_trunc_err_t.dat"))
    push!(ϵDTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_D_trunc_err_t.dat"))
end

n_t_s = Array{Float64}(undef, length(CHIS), length(OMEGAS), nTimeSteps + 1);
n_qs_s = Array{Float64}(undef, length(CHIS), length(OMEGAS));

for i in eachindex(CHIS)
    for (j, OMEGA) in enumerate(OMEGAS)
        FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHIS[i])_K_$(KRAUSDIMS[i])_$(JOBID)"
        n_t_s[i, j, :] = deserialize(RESULT_PATH * FILE_INFO * "_n_t.dat")
        n_qs_s[i, j] = n_t_s[i, j, end]
    end
end

# # n_qs
aplot = plot();
ytick_positions = [1.0, 0.1, 0.01]
ytick_labels = ["1.0", "0.1", "0.01"]
FILE = "N_$(N)_dt_$(dt)_ntime_$(nTimeSteps)_$(JOBID)";

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
                ls=(if i == 1
                    :dashdot
                elseif i == 2
                    :dash
                else
                    :solid
                end),
                lc=(if OMEGA == 1.9
                    colorPal[1]
                elseif OMEGA == 5.7
                    colorPal[2]
                else
                    colorPal[3]
                end),
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

## ϵHTrunc
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

## ϵDTrunc
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

## entanglement entropy
# aplot = plot();
# for (i, OMEGA) in enumerate(OMEGAS)
#     vNEnts = [computevNEntropy(x) for x in ent_spec_t[i]]

#     plot!(aplot, dt*(1:nTimeSteps), vNEnts, label=L"\Omega = %$OMEGA")
# end

# plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Half-chain~von~Neumann~entropy}",
#     #   xaxis=:log10,
#       title=L"N=%$N")
# savefig(aplot, OUTPUT_PATH * FILES * "_vnEnt_t.pdf")

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
