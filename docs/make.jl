using Documenter, SolverTest

makedocs(
  modules = [SolverTest],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
  ),
  sitename = "SolverTest.jl",
  pages = ["Home" => "index.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/SolverTest.jl.git",
  devbranch = "main",
  push_preview = true,
)
