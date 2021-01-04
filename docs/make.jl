push!(LOAD_PATH,"../src/")
using ArDCA
using Documenter

makedocs(;
    modules=[ArDCA],
    authors="Andrea Pagnani, Jeanne Trinquier, Guido Uguzzoni, Martin Weigt, Francesco Zamponi",
    repo="https://github.com/pagnani/ArDCA/blob/{commit}{path}#L{line}",
    sitename="ArDCA",
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
    repo="github.com/pagnani/ArDCA",
)
