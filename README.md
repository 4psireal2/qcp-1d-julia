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
- Current findings / questions:
    - Bad convergence for N > 5 -> Due to the gap of L̂^† L̂ and degeneracy?
    - A mixed left/right isometric form for DMRG? -> [Publication](https://doi.org/10.1103/PhysRevB.87.155137)
    - Plot L̂^† L̂ spectrum for small N?
    - Note on III. Physical solutions (Supplementary Material)
    - Transfer matrix
- Checked:
    - Hermiticity  of rho, numberOp
- TODO:
    - Reproduce Ising Chain with coherent dissipation (benchmarking model)
    - Construct  L̂^† L̂ and compute EVal spectrum
    - modularize my code