## Local development
- Format code

    make format

- Run script from root directory

    julia --project=. path/to/script.jl

- Run a script interactively with Julia REPL (Read-Eval-Print Loop) in a virutalenv

    ]

    activate .

    Ctrl + c

    include("path/to/script.jl")

- Add dependencies / packages

    ]

    activate .
    
    add <package>