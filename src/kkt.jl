using QuadraticModels, RipQP, SparseArrays

export kkt_checker

"""
    kkt_checker(nlp, sol; kwargs...)

Given an NLPModels `nlp` and a vector `sol`, it returns the KKT residual of an optimization problem as a tuple (primal, dual).
In particular, it uses `ripqp` to solve the following quadratic optimization problem with linear constraints
```
min_{d} ∇f(sol)ᵀd +  ½∥d∥²
        lvar ≤ sol + d ≤ uvar
        lcon ≤ c(sol) + ∇c(sol)d ≤ ucon
```
The solution of this problem is the gradient of the Lagrangian of the `nlp` at `sol` thanks to the ½ ‖d‖² term in the objective.

Keyword arguments are passed to `RipQP`.
"""
function kkt_checker(
  nlp::AbstractNLPModel{T, S},
  sol;
  kwargs...,
) where {T, S}
  nvar = nlp.meta.nvar
  g = grad(nlp, sol)
  Hrows, Hcols = collect(1:nvar), collect(1:nvar)
  Hvals = ones(T, nlp.meta.nvar)

  feas_res = max.(nlp.meta.lvar - sol, sol - nlp.meta.uvar, 0)
  kkt_nlp = if nlp.meta.ncon > 0
    c = cons(nlp, sol)
    feas_res = vcat(max.(nlp.meta.lcon - c, c - nlp.meta.ucon, 0), feas_res)
    Arows, Acols = jac_structure(nlp)
    Avals = jac_coord(nlp, sol)
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      Arows = Arows,
      Acols = Acols,
      Avals = Avals,
      lcon = nlp.meta.lcon .- c,
      ucon = nlp.meta.ucon .- c,
      lvar = nlp.meta.lvar .- sol,
      uvar = nlp.meta.uvar .- sol,
      x0 = fill!(S(undef, nlp.meta.nvar), zero(T)),
    )
  else
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      lvar = nlp.meta.lvar .- sol,
      uvar = nlp.meta.uvar .- sol,
      x0 = fill!(S(undef, nlp.meta.nvar), zero(T)),
    )
  end
  stats = ripqp(kkt_nlp; display = false, kwargs...)
  if !(stats.status ∈ (:acceptable, :first_order))
    @warn "Failure in the Lagrange multiplier computation, the status of ripqp is $(stats.status)."
  end
  dual_res = stats.solution
  return feas_res, dual_res
end
