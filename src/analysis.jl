using Plots
using ColorSchemes
using Serialization
using LaTeXStrings


N = 11;
nTimeSteps = 50;
dt = 0.10; # 0.01 ≤ dt ≤ 0.1
OMEGAS = [2.0, 6.0, 10.0];
GAMMA = 1.0;

OUTPUT_PATH = "/home/psireal42/study/qcp-1d-julia/outputs/"


# load files
n_t_s = Array{Float64}(undef, length(OMEGAS), nTimeSteps + 1);

for (i, OMEGA) in enumerate(OMEGAS)
    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_K_$(KRAUSDIM)_82_occ";
    n_t_s[i, :] = deserialize(OUTPUT_PATH * FILE_INFO * "_n_t.dat");
end

aplot = plot();
for (i, OMEGA) in enumerate(OMEGAS)
    plot!(aplot, dt*(1:nTimeSteps+1), n_t_s[i, :], label=L"\Omega = %$OMEGA")
end


savefig(aplot, OUTPUT_PATH * "av_n(t).png")

