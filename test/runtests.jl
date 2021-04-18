# stdlib
using LinearAlgebra, Test
# JSO
using NLPModels, OptSolver, SolverCore, SolverTest, SolverTools

include("dummy-solver.jl")

@testset "Testing dummy-solver" begin
  @testset "$foo" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
    unconstrained_nls,
    bound_constrained_nls,
    equality_constrained_nls,
  ]
    foo(DummySolver)
  end


  @testset "Multiprecision tests" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(DummySolver, ptype)
    end

    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nls(DummySolver, ptype)
    end
  end
end