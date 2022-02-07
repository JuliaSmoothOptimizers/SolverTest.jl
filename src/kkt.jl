using LLSModels, RipQP, SparseArrays

export kkt_checker

"""
    kkt_checker(nlp, sol; bound_tol=sqrt(eps(T)), feas_tol=sqrt(eps(T)), kwargs...)

Given an NLPModels `nlp` and a vector `sol`, it returns the KKT residual of an optimization problem as a tuple (primal, dual).
In particular, it uses `ripqp` to solve the following bound-constrained linear least squares:
```
min_{λ, μ} ½∥ μ + ∇c(x)ᵀλ + ∇f(x)∥²
            μᵢ = 0 for i s.t. ℓᵢ < xᵢ < uᵢ
            μᵢ ≥ 0 for i s.t. xᵢ = uᵢ > ℓᵢ
            μᵢ ≤ 0 for i s.t. xᵢ = ℓᵢ < uᵢ
            μᵢ no constraints for i s.t. xᵢ = ℓᵢ = uᵢ
            λᵢ = 0 for i s.t. lconᵢ < cᵢ(x) < uconᵢ
            λᵢ ≥ 0 for i s.t. cᵢ(x) = uconᵢ > lconᵢ
            λᵢ ≤ 0 for i s.t. cᵢ(x) = lconᵢ < uconᵢ
            λᵢ no constraints for i s.t. cᵢ(x) = lconᵢ = uconᵢ
```

`bound_tol` and `feas_tol` are respectively the tolerances to consider a bound or a constraint active.
Other keyword arguments are passed to `RipQP`.
"""
function kkt_checker(
  nlp::AbstractNLPModel{T, S},
  sol;
  bound_tol=sqrt(eps(T)),
  feas_tol=sqrt(eps(T)),
  kwargs...,
) where {T, S}
  nμ = nlp.meta.nvar
  nλ = nlp.meta.ncon
  Iμfree, Iμfix, Iμp, Iμm = _split_indices(nlp.meta.lvar, nlp.meta.uvar, sol, bound_tol)
  lvar = -T(Inf) * ones(T, nμ + nλ)
  uvar = T(Inf) * ones(T, nμ + nλ)
  lvar[Iμfree] .= zero(T)
  lvar[Iμp] .= zero(T)
  uvar[Iμfree] .= zero(T)
  uvar[Iμm] .= zero(T)

  cx = cons(nlp, sol)
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, sol)

  if !unconstrained(nlp)
    Iλfree, Iλfix, Iλp, Iλm = _split_indices(nlp.meta.lcon, nlp.meta.ucon, cx, feas_tol)
    lvar[Iλfree .+ nμ] .= zero(T)
    lvar[Iλp .+ nμ] .= zero(T)
    uvar[Iλfree .+ nμ] .= zero(T)
    uvar[Iλm .+ nμ] .= zero(T)
  end

  A = sparse(vcat(1:nμ, cols), vcat(1:nμ, rows .+ nμ), vcat(1:nμ, vals), nμ, nμ + nλ)
  b = grad(nlp, sol)
  kkt_nlp = LLSModel(A, b, lvar=lvar, uvar=uvar)
  stats = ripqp(kkt_nlp, display = false, kwargs...)
  if !(stats.status ∈ (:acceptable, :first_order))
    @warn "Failure in the Lagrange multiplier computation, the status of ripqp is $(stats.status)."
  end
  dual_res = residual(kkt_nlp, stats.solution)
  
  feas_res = max.(nlp.meta.lcon - cx, cx - nlp.meta.ucon, 0)
  # μ, λ = stats.solution[1:nμ], stats.solution[nμ + 1:nμ + nλ]
  return feas_res, dual_res
end

# if setdiff(union(Iλfree, Iλfix, Iλp, Iλm), 1:nλ) != [] then `sol` doesn't satisfy the constraints
function _split_indices(lvar, uvar, sol, tol)
  Iμfree = findall(lvar .+ tol .< sol .< uvar .- tol)
  Iμfix = findall(isapprox.(lvar, sol, atol=tol) .&  isapprox.(uvar, sol, atol=tol))
  Iμp = findall(isapprox.(lvar, sol, atol=tol) .& (sol .< uvar .- tol))
  Iμm = findall(isapprox.(uvar, sol, atol=tol) .& (sol .> lvar .+ tol))
  return Iμfree, Iμfix, Iμp, Iμm
end
