module SolverTest

# stdlib
using LinearAlgebra, Logging, SparseArrays, Test
# JSO
using ADNLPModels, NLPModels

include("kkt.jl")

include("nlp/unconstrained.jl")
include("nlp/bound-constrained.jl")
include("nlp/equality-constrained.jl")
include("nlp/multiprecision.jl")

include("nls/unconstrained.jl")
include("nls/bound-constrained.jl")
include("nls/equality-constrained.jl")
include("nls/multiprecision.jl")

end # module
