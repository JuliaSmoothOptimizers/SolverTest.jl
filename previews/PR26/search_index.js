var documenterSearchIndex = {"docs":
[{"location":"#SolverTest","page":"Home","title":"SolverTest","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SolverTest is a package to test JSO-compliant solvers, both for general optimization problems and for nonlinear least-squares problems. It should be pkg> added to [extras] and to the test target in [targets].","category":"page"},{"location":"","page":"Home","title":"Home","text":"The following functions are available:","category":"page"},{"location":"","page":"Home","title":"Home","text":"unconstrained_nlp\nbound_constrained_nlp\nequality_constrained_nlp\nunconstrained_nls\nbound_constrained_nls\nequality_constrained_nls\nmultiprecision_nlp\nmultiprecision_nls","category":"page"},{"location":"#SolverTest.unconstrained_nlp","page":"Home","title":"SolverTest.unconstrained_nlp","text":"unconstrained_nlp(solver; problem_set = unconstrained_nlp_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on unconstrained problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.bound_constrained_nlp","page":"Home","title":"SolverTest.bound_constrained_nlp","text":"bound_constrained_nlp(solver; problem_set = bound_constrained_nlp_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on bound-constrained problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.equality_constrained_nlp","page":"Home","title":"SolverTest.equality_constrained_nlp","text":"equality_constrained_nlp(solver; problem_set = equality_constrained_nlp_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on equality-constrained problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.unconstrained_nls","page":"Home","title":"SolverTest.unconstrained_nls","text":"unconstrained_nls(solver; problem_set = unconstrained_nls_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on unconstrained nonlinear least-squares problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.bound_constrained_nls","page":"Home","title":"SolverTest.bound_constrained_nls","text":"bound_constrained_nls(solver; problem_set = bound_constrained_nls_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on bound-constrained nonlinear least-squares problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.equality_constrained_nls","page":"Home","title":"SolverTest.equality_constrained_nls","text":"equality_constrained_nls(solver; problem_set = equality_constrained_nls_set(), atol = 1e-6, rtol = 1e-6)\n\nTest the solver on equality-constrained problems. If rtol is non-zero, the relative error uses the gradient at the initial guess.\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.multiprecision_nlp","page":"Home","title":"SolverTest.multiprecision_nlp","text":"multiprecision_nlp(solver, problem_type; precisions=[Float16, …, BigFloat])\n\nTest that solver solves a problem of type problem_type on various precisions. The problem_type can be\n\n:unc\n:bnd\n:equ\n:ineq\n:eqnbnd\n:gen\n\n\n\n\n\n","category":"function"},{"location":"#SolverTest.multiprecision_nls","page":"Home","title":"SolverTest.multiprecision_nls","text":"multiprecision_nls(solver, problem_type; precisions=[Float16, …, BigFloat])\n\nTest that solver solves a problem of type problem_type on various precisions. The problem_type can be\n\n:unc\n:bnd\n:equ\n:ineq\n:eqnbnd\n:gen\n\n\n\n\n\n","category":"function"},{"location":"#Auxiliary-funcions","page":"Home","title":"Auxiliary funcions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SolverTest.kkt_checker","category":"page"},{"location":"#SolverTest.kkt_checker","page":"Home","title":"SolverTest.kkt_checker","text":"kkt_checker(nlp, sol; bound_tol=sqrt(eps(T)), feas_tol=sqrt(eps(T)), kwargs...)\n\nGiven an NLPModels nlp and a vector sol, it returns the KKT residual of an optimization problem as a tuple (primal, dual). In particular, it uses ripqp to solve the following bound-constrained linear least squares:\n\nmin_{λ, μ} ½∥ μ + ∇c(x)ᵀλ + ∇f(x)∥²\n            μᵢ = 0 for i s.t. ℓᵢ < xᵢ < uᵢ\n            μᵢ ≥ 0 for i s.t. xᵢ = uᵢ > ℓᵢ\n            μᵢ ≤ 0 for i s.t. xᵢ = ℓᵢ < uᵢ\n            μᵢ no constraints for i s.t. xᵢ = ℓᵢ = uᵢ\n            λᵢ = 0 for i s.t. lconᵢ < cᵢ(x) < uconᵢ\n            λᵢ ≥ 0 for i s.t. cᵢ(x) = uconᵢ > lconᵢ\n            λᵢ ≤ 0 for i s.t. cᵢ(x) = lconᵢ < uconᵢ\n            λᵢ no constraints for i s.t. cᵢ(x) = lconᵢ = uconᵢ\n\nbound_tol and feas_tol are respectively the tolerances to consider a bound or a constraint active. Other keyword arguments are passed to RipQP.\n\n\n\n\n\n","category":"function"}]
}