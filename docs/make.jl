push!(LOAD_PATH,"../src/")
using ArDCA
using Documenter

makedocs(;
    modules=[ArDCA],
    authors="Andrea Pagnani, Jeanne Trinquier, Guido Uguzzoni, Martin Weigt, Francesco Zamponi",
    clean=true,
    #repo="https://github.com/pagnani/ArDCA.jl/blob/{commit}{path}#L{line}",
    sitename="ArDCA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pagnani.github.io/ArDCA",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pagnani/ArDCA.jl.git",
)
