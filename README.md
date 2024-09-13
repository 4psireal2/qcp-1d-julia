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

## Implemented tests for finite-size simulations
[x] Finite-T states for TFI model (infinite-T state -> imaginary time evolution f with TEBD to get ground state) >< DMRG code of TeNPy
[x] TEBD + LPTN for QCP model >< ED (single-site observable)
