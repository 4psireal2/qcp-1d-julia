# setup_env.jl
using Pkg
Pkg.activate(ENV["JULIA_PROJECT"])
Pkg.instantiate()
