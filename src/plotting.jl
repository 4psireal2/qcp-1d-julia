using NumericIO
using Plots
using Printf
using ColorSchemes
using Serialization
using LaTeXStrings

default(fontfamily="Palatino Roman")


N = 10;
nTimeSteps = 50;
dt = 0.1; # 0.01 ≤ dt ≤ 0.1
JOBID = 508781;
CHI = 50;
KRAUSDIM = 25;

OMEGAS = [2.0, 6.0, 10.0];
GAMMA = 1.0;

RESULT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/results/"
OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/hpc/outputs/"
FILES = "N_$(N)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";


# asciiexponentfmt = NumericIO.IOFormattingExpNum(
# 	"x10^", false, '+', '-', NumericIO.ASCII_SUPERSCRIPT_NUMERALS
# )
# fmt = NumericIO.IOFormattingReal(asciiexponentfmt,
# 	ndigits=2, decpos=0, decfloating=true, eng=true, minus='-', inf="Inf"
# )


## load files
n_t_s = Array{Float64}(undef, length(OMEGAS), nTimeSteps + 1);
# n_t_s_short = Array{Float64}(undef, length(OMEGAS), nTimeSteps + 1);

ϵHTrunc_t = Any[];
ϵDTrunc_t = Any[];

for (i, OMEGA) in enumerate(OMEGAS)
    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(CHI)_K_$(KRAUSDIM)_$(JOBID)";

    n_t_s[i, :] = deserialize(RESULT_PATH * FILE_INFO * "_n_t.dat");
    # n_t_s_short[i, :] = deserialize(RESULT_PATH * FILE_INFO * "_n_t_short.dat");

    push!(ϵHTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_H_trunc_err_t.dat"))
    push!(ϵDTrunc_t, deserialize(RESULT_PATH * FILE_INFO * "_D_trunc_err_t.dat"))

end

## av(t)
aplot = plot();
ytick_positions = [1.0, 0.1, 0.01]
ytick_labels = ["1.0", "0.1", "0.01"]

for (i, OMEGA) in enumerate(OMEGAS)
    plot!(dt*(1:nTimeSteps+1), n_t_s[i, :], label=L"\Omega = %$OMEGA") 
end
plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{n(t)}",
          title=L"\chi=%$CHI, \, K=%$KRAUSDIM, \, N=%$N",
          legend=:bottomleft,
          xaxis=:log10, yaxis=:log10,
          yticks=(ytick_positions, ytick_labels),
          xformatter = (val) -> @sprintf("%.1f", val), yformatter = (val) -> @sprintf("%.2f", val))
savefig(aplot, OUTPUT_PATH * FILES * "_av_n(t).pdf")

## av(t) short
# aplot = plot();
# ytick_positions = [1.0, 0.1, 0.01]
# ytick_labels = ["1.0", "0.1", "0.01"]

# for (i, OMEGA) in enumerate(OMEGAS)
#     plot!(dt*(1:nTimeSteps+1), n_t_s_short[i, :], label=L"\Omega = %$OMEGA") 
# end
# plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{n(t)}",
#           title=L"\chi=%$CHI, \, K=%$KRAUSDIM, \, N=%$N",
#           legend=:bottomleft,
#           xaxis=:log10, yaxis=:log10,
#           yticks=(ytick_positions, ytick_labels),
#           xformatter = (val) -> @sprintf("%.1f", val), yformatter = (val) -> @sprintf("%.2f", val))
# savefig(aplot, OUTPUT_PATH * FILES * "_av_n(t)_short.pdf")

## ϵHTrunc
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_sum =  [sum(x) for x in ϵHTrunc_t[i]];
    ϵHTrunc_t_max =  [maximum(x) for x in ϵHTrunc_t[i]];
    plot!(aplot, dt*(1:nTimeSteps), ϵHTrunc_t_sum, label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}")
    plot!(aplot, dt*(1:nTimeSteps), ϵHTrunc_t_max, label=L"\Omega = %$OMEGA, \, \textrm{max. err.}") 
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Truncation~error~in~} \chi, \, \chi=%$CHI",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_t.pdf")

## cumulative ϵHTrunc accum.
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_sum =  [sum(x) for x in ϵHTrunc_t[i]];
    ϵHTrunc_t_cumsum =  cumsum(ϵHTrunc_t_sum) / (nTimeSteps * dt);

    plot!(aplot, dt*(1:nTimeSteps), ϵHTrunc_t_cumsum, label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}")
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Cumulative~truncation~error~in~} \chi, \, \chi=%$CHI",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_cumsum_sum_t.pdf")

## cumulative ϵHTrunc max
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵHTrunc_t_max =  [maximum(x) for x in ϵHTrunc_t[i]];
    ϵHTrunc_t_cumsum_max =  cumsum(ϵHTrunc_t_max) / (nTimeSteps * dt);

    plot!(aplot, dt*(1:nTimeSteps), ϵHTrunc_t_cumsum_max, label=L"\Omega = %$OMEGA, \, \textrm{max. err.}")
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Cumulative~truncation~error~in~} \chi, \, \chi=%$CHI",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_H_trunc_err_cumsum_sum_max_t.pdf")

## ϵDTrunc
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_sum =  [sum(x) for x in ϵDTrunc_t[i]];
    ϵDTrunc_t_max =  [maximum(x) for x in ϵDTrunc_t[i]];
    plot!(aplot, dt*(1:nTimeSteps), ϵDTrunc_t_sum, label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}")
    plot!(aplot, dt*(1:nTimeSteps), ϵDTrunc_t_max, label=L"\Omega = %$OMEGA, \, \textrm{max. err.}") 
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Truncation~error~in~} K, \, K=%$KRAUSDIM",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_t.pdf")

## cumulative ϵDTrunc accum.
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_sum =  [sum(x) for x in ϵDTrunc_t[i]];
    ϵDTrunc_t_cumsum =  cumsum(ϵDTrunc_t_sum) / (nTimeSteps * dt);

    plot!(aplot, dt*(1:nTimeSteps), ϵDTrunc_t_cumsum, label=L"\Omega = %$OMEGA, \, \textrm{acc. err.}")
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Cumulative~truncation~error~in~} K, \, K=%$KRAUSDIM",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_cumsum_sum_t.pdf")

## cumulative ϵDTrunc max
aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    ϵDTrunc_t_max =  [maximum(x) for x in ϵDTrunc_t[i]];
    ϵDTrunc_t_cumsum_max =  cumsum(ϵDTrunc_t_max) / (nTimeSteps * dt);

    plot!(aplot, dt*(1:nTimeSteps), ϵDTrunc_t_cumsum_max, label=L"\Omega = %$OMEGA, \, \textrm{max. err.}")
end

plot!(xlabel=L"\textrm{t}", ylabel=L"\textrm{Cumulative~truncation~error~in~} K, \, K=%$KRAUSDIM",
          title=L"N=%$N")
savefig(aplot, OUTPUT_PATH * FILES * "_D_trunc_err_cumsum_sum_max_t.pdf")

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