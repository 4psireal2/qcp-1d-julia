"""
Dynamical simulations of QCP for short chain N = 11
"""

using Pkg
Pkg.activate(ENV["JULIA_PROJECT"])

using ArgParse
using Logging
using Serialization

include("../src/lptn.jl")
include("../src/models.jl")
include("../src/tebd.jl")

OUTPUT_PATH = "/scratch/nguyed99/qcp-1d-julia/results/";
LOG_PATH = "/scratch/nguyed99/qcp-1d-julia/logging/"

# TN algorithm parameters
truncErr = 1e-6;

# observable operator
basis0 = [1, 0];
basis1 = [0, 1];
numberOp = [0 0; 0 1];
numberOp = TensorMap(numberOp, ℂ^2, ℂ^2);

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--N"
        help = "System size"
        arg_type = Int

        "--OMEGA"
        help = "Branching rate"
        arg_type = Float64

        "--BONDDIM"
        help = "Bond dimension"
        arg_type = Int

        "--KRAUSDIM"
        help = "Kraus dimension"
        arg_type = Int

        "--dt"
        help = "Time step size"
        arg_type = Float64

        "--nt"
        help = "Number of time steps"
        arg_type = Int

        "--JOBID"
        help = "SLURM array job ID"
        arg_type = Int
    end

    parsed_args = parse_args(args, s)
    N = parsed_args["N"]
    OMEGA = parsed_args["OMEGA"]
    BONDDIM = parsed_args["BONDDIM"]
    KRAUSDIM = parsed_args["KRAUSDIM"]
    dt = parsed_args["dt"]
    nTimeSteps = parsed_args["nt"]
    SLURM_ARRAY_JOB_ID = parsed_args["JOBID"]
    GAMMA = 1.0

    FILE_INFO = "N_$(N)_OMEGA_$(OMEGA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(BONDDIM)_K_$(KRAUSDIM)_$(SLURM_ARRAY_JOB_ID)"
    logFile = open(LOG_PATH * "$(FILE_INFO).log", "w+")
    logger = SimpleLogger(logFile, Logging.Info)
    Base.global_logger(logger)

    @info "System and simulation info: N=$N, OMEGA=$OMEGA, GAMMA=$GAMMA, BONDDIM=$BONDDIM, KRAUSDIM=$KRAUSDIM with truncErr=$truncErr"

    n_t = zeros(nTimeSteps + 1)
    ϵHTrunc_t = Vector{Float64}[]
    ϵDTrunc_t = Vector{Float64}[]
    entEntropy_t = []
    renyiEnt_t = [];
    densDensCorrelation = Array{Float64}(undef, N - 1)
    n_sites_t = Array{Float64}(undef, nTimeSteps + 1, N)

    basisTogether = vcat(fill([basis0, basis1, basis1, basis1, basis1], N ÷ 5)...)
    XInit = createXBasis(N, basisTogether)
    # XInit = createXBasis(N, [fill(basis0, 4); [Vector{Int64}(basis1)]; fill(basis0, 5)]);

    n_sites_t[1, :], n_t[1] = computeSiteExpVal!(XInit, numberOp)
    XInit = orthonormalizeX!(XInit; orthoCenter=1)

    @info "Initial average number of particles: $(n_t[1])"

    @info "Running second-order TEBD..."
    elapsed_time = @elapsed begin
        hamDyn = expHam(OMEGA, dt / 2)
        dissDyn = expDiss(GAMMA, dt)
        X_t = XInit
        for i in 1:nTimeSteps
            @time allocated_tebd = @allocated X_t, ϵHTrunc, ϵDTrunc = TEBD(
                X_t,
                hamDyn,
                dissDyn,
                BONDDIM,
                KRAUSDIM;
                truncErr=truncErr,
                canForm=true, # orthonormalized X_t
            )
            @info "TEBD done for $(i)-th time step. Allocated memory: $(allocated_tebd/2^30) GB"
            flush(logFile)

            if i == nTimeSteps
                @time allocated_corr = @allocated begin
                    for j in 2:N
                        densDensCorrelation[j - 1] = densDensCorr(j, X_t, numberOp)
                    end
                end
                @info "Density-Density Correlation calculated. Allocated memory: $(allocated_corr/2^30) GB"
            end

            @time allocated_sites = @allocated n_sites_t[i + 1, :], n_t[i + 1] = computeSiteExpVal!(
                X_t, numberOp
            ) # right-canonical
            @info "Site Expectation Value computed. Allocated memory: $(allocated_sites/2^30) GB"

            push!(ϵHTrunc_t, ϵHTrunc)
            push!(ϵDTrunc_t, ϵDTrunc)

            @time allocated_entEntropy = @allocated entEntropy = computeEntEntropy!(X_t) # mid-canonical
            push!(entEntropy_t, entEntropy)
            @info "Entanglement entropy computed. Allocated memory: $(allocated_entEntropy/2^30) GB"

            @time allocated_renyi = @allocated renyi_ent = compute2RenyiEntropy!(X_t)
            push!(renyiEnt_t, renyi_ent)
            @info "2-Renyi Entropy computed. Allocated memory: $(allocated_renyi/2^30) GB"

            @info "Computation of properties done for $(i)-th time step"
            flush(logFile)

            X_t = orthonormalizeX!(X_t; orthoCenter=1)
        end
    end

    @info "Elapsed time for TEBD: $elapsed_time seconds"
    @info "Saving data..."

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

    open(OUTPUT_PATH * FILE_INFO * "_ent_entropy_t.dat", "w") do file
        serialize(file, entEntropy_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_renyiEnt_t.dat", "w") do file
        serialize(file, renyiEnt_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_dens_dens_corr.dat", "w") do file
        serialize(file, densDensCorrelation)
    end

    return close(logFile)
end

main(ARGS)
