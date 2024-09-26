"""
Dynamical simulations of dissipative XXZ chain
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
basis = 1 / (sqrt(2)) * (basis0 + basis1);
Sz = [+1 0; 0 -1];
Sz = TensorMap(Sz, ℂ^2, ℂ^2);

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--N"
        help = "System size"
        arg_type = Int

        "--DELTA"
        help = "Sz interaction rate"
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
    DELTA = parsed_args["DELTA"]
    BONDDIM = parsed_args["BONDDIM"]
    KRAUSDIM = parsed_args["KRAUSDIM"]
    dt = parsed_args["dt"]
    nTimeSteps = parsed_args["nt"]
    SLURM_ARRAY_JOB_ID = parsed_args["JOBID"]
    GAMMA = 1.0

    FILE_INFO = "N_$(N)_DELTA_$(DELTA)_dt_$(dt)_ntime_$(nTimeSteps)_CHI_$(BONDDIM)_K_$(KRAUSDIM)_$(SLURM_ARRAY_JOB_ID)"
    logFile = open(LOG_PATH * "$(FILE_INFO).log", "w+")
    logger = SimpleLogger(logFile, Logging.Info)
    Base.global_logger(logger)

    @info "System and simulation info: N=$N, DELTA=$DELTA, GAMMA=$GAMMA, BONDDIM=$BONDDIM, KRAUSDIM=$KRAUSDIM with truncErr=$truncErr"

    Sz_t = zeros(nTimeSteps + 1)
    ϵHTrunc_t = Vector{Float64}[]
    ϵDTrunc_t = Vector{Float64}[]
    Sz_sites_t = Array{Float64}(undef, nTimeSteps + 1, N)

    # basisTogether = vcat(fill([basis0, basis0, basis0, basis0, basis0], N ÷ 5)...)
    basisTogether = vcat(fill([basis0], N ÷ 2)..., fill([basis1], N ÷ 2)...)
    XInit = createXBasis(N, basisTogether)

    Sz_sites_t[1, :], Sz_t[1] = computeSiteExpVal!(XInit, Sz)
    XInit = orthonormalizeX!(XInit; orthoCenter=1)

    @info "Running second-order TEBD..."
    elapsed_time = @elapsed begin
        hamDyn = expXXZHam(DELTA, dt / 2)
        dissDynL = expXXZDissL(GAMMA, dt)
        dissDynR = expXXZDissR(GAMMA, dt)

        X_t = XInit
        for i in 1:nTimeSteps
            @time allocated_tebd = @allocated X_t, ϵHTrunc, ϵDTrunc = TEBD_boundary(
                X_t,
                hamDyn,
                dissDynL,
                dissDynR,
                BONDDIM,
                KRAUSDIM;
                truncErr=truncErr,
                canForm=true, # orthonormalized X_t
            )
            @info "TEBD done for $(i)-th time step. Allocated memory: $(allocated_tebd/2^30) GB"
            flush(logFile)

            @time allocated_sites = @allocated Sz_sites_t[i + 1, :], Sz_t[i + 1] = computeSiteExpVal!(
                X_t, Sz
            ) # right-canonical
            @info "Site Expectation Value computed. Allocated memory: $(allocated_sites/2^30) GB"

            push!(ϵHTrunc_t, ϵHTrunc)
            push!(ϵDTrunc_t, ϵDTrunc)

            @info "Computation of properties done for $(i)-th time step"
            flush(logFile)

            X_t = orthonormalizeX!(X_t; orthoCenter=1)
        end
    end

    @info "Elapsed time for TEBD: $elapsed_time seconds"
    @info "Saving data..."

    open(OUTPUT_PATH * FILE_INFO * "_Sz_sites_t.dat", "w") do file
        serialize(file, Sz_sites_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_Sz_t.dat", "w") do file
        serialize(file, Sz_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_H_trunc_err_t.dat", "w") do file
        serialize(file, ϵHTrunc_t)
    end

    open(OUTPUT_PATH * FILE_INFO * "_D_trunc_err_t.dat", "w") do file
        serialize(file, ϵDTrunc_t)
    end

    return close(logFile)
end

main(ARGS)
