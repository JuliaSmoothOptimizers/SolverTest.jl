# Tutorial

The package SolverTest provides a set a functions to test the basic features of a JSO-compliant solver. 
It is assumed that:
1. the tested solver accepts an [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) as input, and;
2. returns a [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) as output;
3. The solver function should accepts `atol` and `rtol` as keyword arguments.

## Unconstrained Optimization

The function [`unconstrained_nlp_set`](@doc) returns a list of optimization problems in the [`ADNLPModel`](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) format.

Secondly, the function [`unconstrained_nlp`](@doc) will test the behavior of the JSO-compliant function `solver` on the test set.
More precisely, it will test that primal and dual feasibility at the solution (`stats.solution`) computed with [`kkt_checker`](@doc) and that `stats.dual_feas` are relatively small, i.e. ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖ where atol = rtol = 1e-6.
It will also be tested that the final status `stats.status` is `:first_order`.
The problems are sufficiently simple that we expect the solver to solve them successfully.

Similarly, [`unconstrained_nls_set`](@doc) and [`unconstrained_nls`](@doc) are preparing the same tests for nonlinear least squares.

```@example
using SolverTest
using Test
using JSOSolvers # export solver lbfgs
@testset "Unconstrained solvers" begin
  unconstrained_nlp(lbfgs)
  multiprecision_nlp(lbfgs, :unc)
end
```

## Bound-constrained Optimization

## Equality-constrained Optimization

## Bound and Equality-constrained Optimization

This is still work in progress, see [Issue 8](https://github.com/JuliaSmoothOptimizers/SolverTest.jl/issues/8).

## Inequality-constrained optimization

This is still work in progress, see [Issue 8](https://github.com/JuliaSmoothOptimizers/SolverTest.jl/issues/8).

## Multi-precision tests

TODO
