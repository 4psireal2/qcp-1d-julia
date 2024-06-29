"""
Dynamical simulations of QCP for short chain
"""

include("../src/lptn.jl")
include("../src/tebd.jl")

using Plots
using ColorSchemes

N = 7
nTimeSteps = 30;
dt = 0.05; # 0.01 ≤ dt ≤ 0.1
OMEGA = 3.0;
GAMMA = 1.0;

truncErr = 1e-6;
BONDDIM = 15;
KRAUSDIM = 15;

basis0 = [1, 0];
basis1 = [0, 1];
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);


n_t = zeros(nTimeSteps + 1);
n_sites_t = Array{Float64}(undef, nTimeSteps+1, N);
XInit = createXBasis(N, [fill(basis0, N ÷ 2); [Vector{Int64}(basis1)]; fill(basis0, N ÷ 2)]);
n_sites_t[1, :], n_t[1] = computeSiteExpVal(XInit, numberOp);
println("Initial average number of particles: $(n_t[1])")

# run dynamics simulation
hamDyn = expHam(OMEGA, dt/2);
dissDyn = expDiss(GAMMA, dt);
X_t = XInit;
for i = 1 : nTimeSteps
    global  X_t
    X_t = TEBD(X_t, hamDyn, dissDyn, BONDDIM, KRAUSDIM, truncErr);

    n_sites_t[i+1, :], n_t[i+1] = computeSiteExpVal(X_t, numberOp);
end


heatmap(reverse(n_sites_t, dims=1), colorbar=true, colormap=:Greys, yticks=(1:5:nTimeSteps+1, nTimeSteps+1:-5:1), xlabel="Time", ylabel="Average number of particles")
savefig("N_7_OMEGA_3_CHI_15.png")