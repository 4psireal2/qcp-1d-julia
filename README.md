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
    - Start with a small χ and use the result as initial guess for the search with increased χ.
    - Do not start with a random vector but a suitable initial state.
    - If after the algorithm has converged for χ = 1, if the solution is not compatible with a physical state, the procedure is repeated from a different initial state. Otherwise, the bond dimension is increased, and the obtained state is used as initial guess.
    - Physical solution: the local number operator have expectation values within the physical range [0, 0.5] .