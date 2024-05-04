## Local development
- Format code
```
    make format
```
- Run script from root directory
```
    julia --project=. path/to/script.jl
```
- Run a script interactively with Julia REPL (Read-Eval-Print Loop) in a virutalenv
```
    ]
    activate .
    Ctrl + c
    include("path/to/script.jl")
```

- Add dependencies / packages
```
    ]
    activate .
    add <package>
```

## Open questions / TODO
- Warm-up phase:
    - Goal: Avoid solutions with vanishing trace (zero-MPS / zero MPO) + Improve the convergence of the algorithm 
    - Start with a small χ and use the result as initial guess for the search with increased χ.
    - Do not start with a random vector but a suitable initial state.
    - If after the algorithm has converged for χ = 1, if the solution is not compatible with a physical state, the procedure is repeated from a different initial state. Otherwise, the bond dimension is increased, and the obtained state is used as initial guess.
    - Physical solution: the local number operator has expectation values within the physical range [0, 2]
- Convergence:
    -  relative variation of the energy from one value of the bond dimension to the next. Instead of its relative
       variation, we check the absolute value, so we require that the found eigenvalue is below some threshold, and additionally, demand convergence of some physical observables. 
- Current findings / questions:
    - Bad convergence for N > 5 -> Due to the gap of L̂^† L̂ and degeneracy?
    - A mixed left/right isometric form for DMRG? -> [Publication](https://doi.org/10.1103/PhysRevB.87.155137)
    - Plot L̂^† L̂ spectrum for small N?
    - Note on III. Physical solutions (Supplementary Material)
    - maxIterations = variable for Eval Solver
    - CP with DMRG1
        + N=10, bondDim=16 -> E/N = 0.019969
        + N=10, bondDim=20 -> E/N = 0.013150
        + N=10, bondDim=32 -> E/N = 0.005064
        + N=10, bondDim=40 -> E/N = 0.002810
        + N=10, bondDim=50 -> E/N = 0.001455
        + N=10, bondDim=60 -> E/N = 0.000850

    - Ising chain with DMRG1 -> Start from random states
        + N=10, bondDim=16 -> E/N = 0.143694
        + N=10, bondDim=32 -> E/N = 0.143066
        + N=10, bondDim=40 -> E/N = 0.143038
        + N=10, bondDim=50 -> E/N = 0.143026

- Checked:
    - Hermiticity  of rho, numberOp
- TODO:
    - Reproduce Ising Chain with coherent dissipation (benchmarking model)
    - Construct  L̂^† L̂ and compute EVal spectrum
    - modularize my code