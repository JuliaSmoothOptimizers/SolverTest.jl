using Documenter, SolverTest

makedocs(
  modules = [SolverTest],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
  ),
  sitename = "SolverTest.jl",
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "References" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/SolverTest.jl.git",
  devbranch = "main",
  push_preview = true,
)
