# Tutorial

The package SolverTest provides a set a functions to test the basic features of a JSO-compliant solver. 
It is assumed that:
1. the tested solver accepts an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) as input, and;
2. returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) as output;
3. The solver function should accepts `atol` and `rtol` as keyword arguments.

## Unconstrained Optimization

The function `unconstrained_nlp_set`  returns a list of optimization problems in the [`ADNLPModel`](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) format.

Secondly, the function [`unconstrained_nlp`](@ref) will test the behavior of the JSO-compliant function `solver` on the test set.
More precisely, it will test that primal and dual feasibility at the solution (`stats.solution`) computed with [`kkt_checker`](@ref) and that `stats.dual_feas` are relatively small, i.e. ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖ where atol = rtol = 1e-6.
It will also be tested that the final status `stats.status` is `:first_order`.
The problems are sufficiently simple that we expect the solver to solve them successfully.

```@example
using SolverTest
using Test
using JSOSolvers # export solver trunk
@testset "Unconstrained solvers" begin
  unconstrained_nlp(trunk)
  multiprecision_nlp(trunk, :unc)
end
```

Similarly, `unconstrained_nls_set` and [`unconstrained_nls`](@ref) are preparing the same tests for nonlinear least squares.

```@example
using SolverTest
using Test
using JSOSolvers # export solver trunk
@testset "Unconstrained solvers" begin
  unconstrained_nls(trunk)
  multiprecision_nls(trunk, :unc)
end
```

## Bound-constrained Optimization

The function `bound_constrained_nlp_set`  returns a list of optimization problems in the [`ADNLPModel`](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) format.

Secondly, the function [`bound_constrained_nlp`](@ref) will test the behavior of the JSO-compliant function `solver` on the test set.
More precisely, it will test that primal and dual feasibility at the solution (`stats.solution`) computed with [`kkt_checker`](@ref) and that `stats.primal_feas` and `stats.dual_feas` are relatively small, i.e. smaller than atol + rtol * ‖∇f(x⁰)‖ where atol = rtol = 1e-6.
It will also be tested that the final status `stats.status` is `:first_order`.
The problems are sufficiently simple that we expect the solver to solve them successfully.

```@example
using SolverTest
using Test
using JSOSolvers # export solver trunk
@testset "Bound-constrained solvers" begin
  bound_constrained_nlp(tron)
  multiprecision_nlp(tron, :unc)
  multiprecision_nlp(tron, :bnd)
end
```

Similarly, `bound_constrained_nls_set` and [`bound_constrained_nls`](@ref) are preparing the same tests for nonlinear least squares.

```@example
using SolverTest
using Test
using JSOSolvers # export solver trunk
@testset "Bound-constrained solvers" begin
  bound_constrained_nls(tron)
  multiprecision_nls(tron, :unc)
  multiprecision_nls(tron, :bnd)
end
```

## Equality-constrained Optimization

```@example
using SolverTest
using Test
using DCISolver # export solver dci
@testset "Testing DCI-solver" begin

  @testset "$foo" for foo in [
    unconstrained_nlp,
    bound_constrained_nlp,
    equality_constrained_nlp,
  ]
    foo(dci)
  end

  @testset "Multiprecision tests" begin
    for ptype in [:unc, :bnd, :equ]
      multiprecision_nlp(
        dci,
        ptype,
        precisions = (Float32, Float64),
      )
    end
  end
end
```

## Bound and Equality-constrained Optimization

This is still work in progress, see [Issue 8](https://github.com/JuliaSmoothOptimizers/SolverTest.jl/issues/8).

## Inequality-constrained optimization

This is still work in progress, see [Issue 8](https://github.com/JuliaSmoothOptimizers/SolverTest.jl/issues/8).

## Multi-precision tests

TODO
