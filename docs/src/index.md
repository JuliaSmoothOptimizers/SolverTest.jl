# SolverTest

SolverTest is a package to test JSO-compliant solvers, both for general optimization problems and for nonlinear least-squares problems.
It should be `pkg> add`ed to `[extras]` and to the `test` target in `[targets]`.

The following functions are available:

```@docs
unconstrained_nlp
bound_constrained_nlp
equality_constrained_nlp
unconstrained_nls
bound_constrained_nls
equality_constrained_nls
multiprecision_nlp
multiprecision_nls
```

## Auxiliary funcions

```@docs
SolverTest.eqn_solution_check
SolverTest.kkt_checker
```
