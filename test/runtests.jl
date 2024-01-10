# stdlib
using LinearAlgebra, Test
# JSO
using NLPModels, SolverCore, SolverTest

@testset "Testing dummy-solver" begin
  @testset "$foo" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
    unconstrained_nls,
    bound_constrained_nls,
    equality_constrained_nls,
  ]
    foo(SolverCore.dummy_solver)
  end

  @testset "Multiprecision tests" begin
    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nlp(SolverCore.dummy_solver, ptype)
    end

    for ptype in [:unc, :bnd, :equ, :ineq, :eqnbnd, :gen]
      multiprecision_nls(SolverCore.dummy_solver, ptype)
    end
  end
end
