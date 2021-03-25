# stdlib
using LinearAlgebra, Test
# JSO
using NLPModels, SolverTest, SolverTools

include("dummy-solver.jl")

@testset "Testing dummy-solver" begin
  @testset "$foo" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    unconstrained_nls,
    bound_constrained_nls,
  ]
    foo(dummy)
  end

  @testset "Multiprecision tests" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(dummy, ptype)
    end

    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nls(dummy, ptype)
    end
  end
end