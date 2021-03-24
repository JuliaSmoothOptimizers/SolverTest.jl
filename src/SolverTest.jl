module SolverTest

# stdlib
using LinearAlgebra, Logging, SparseArrays, Test
# JSO
using ADNLPModels, NLPModels

include("multiprecision.jl")
include("nlp/unconstrained.jl")
include("nlp/bound-constrained.jl")
include("nls/unconstrained.jl")
include("nls/bound-constrained.jl")

end # module
