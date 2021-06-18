function dummy(
  nlp::AbstractNLPModel;
  x = copy(nlp.meta.x0),
  atol = 1e-6,
  rtol = 1e-6,
  max_time = 30.0,
  max_eval = 10000,
  max_iter = 1000,
)
  T = eltype(x)
  status = :unknown
  ℓ, u = T.(nlp.meta.lvar), T.(nlp.meta.uvar)
  x .= clamp.(x, ℓ, u)
  ℓidx, uidx = findall(ℓ .> -Inf), findall(u .< Inf)
  n, m = nlp.meta.nvar, nlp.meta.ncon
  nℓ, nu = length(ℓidx), length(uidx)
  Pℓ, Pu = Matrix(I, n, n)[:, ℓidx], Matrix(I, n, n)[:, uidx]

  iter = 0
  t₀ = time()
  Δt = 0.0
  @info("", x)

  yzls(x) = begin
    A = m > 0 ? jac(nlp, x) : zeros(T, 0, n)
    @info("", eltype(A))
    yz = [A' -Pℓ Pu] \ -grad(nlp, x)
    y, zℓ, zu = yz[1:m], Pℓ * yz[m .+ (1:nℓ)], Pu * yz[m .+ nℓ .+ (1:nu)]
    @info("inner", zℓ, zu)
    for i ∈ ℓidx
      if x[i] > ℓ[i] + atol
        zℓ[i] = 0
      end
    end
    for i ∈ uidx
      if x[i] < u[i] - atol
        zu[i] = 0
      end
    end
    for i ∈ ℓidx ∩ uidx
      zℓ[i], zu[i] = max(zℓ[i] - zu[i], 0), max(zu[i] - zℓ[i], 0)
    end
    return y, max.(zℓ, 0), max.(zu, 0)
  end
  y, zℓ, zu = yzls(x)
  J(x) = m > 0 ? jac(nlp, x) : zeros(T, 0, n)
  c(x) = m > 0 ? cons(nlp, x) - T.(nlp.meta.lcon) : zeros(T, 0)

  dual = norm(grad(nlp, x) + J(x)' * y - zℓ + zu)
  primal = norm(c(x))
  compl = max(norm((x[ℓidx] - ℓ[ℓidx]) .* zℓ[ℓidx]), norm((x[uidx] - u[uidx]) .* zu[uidx]))

  @info("", x, y, zℓ, zu, dual)

  ϵ = atol + rtol * max(dual, primal, compl)

  solved = max(dual, primal, compl) < ϵ
  tired = sum_counters(nlp) ≥ max_eval > 0 || Δt ≥ max_time > 0 || iter ≥ max_iter > 0

  τ = T(0.9)
  while !(solved || tired)
    x .= τ * x .+ (1 - τ)
    τ = τ^2

    y, zℓ, zu = yzls(x)
    dual = norm(grad(nlp, x) + J(x)' * y - zℓ + zu)
    primal = norm(c(x))
    compl = max(norm((x[ℓidx] - ℓ[ℓidx]) .* zℓ[ℓidx]), norm((x[uidx] - u[uidx]) .* zu[uidx]))
    @info("", x, y, zℓ, zu, dual)

    iter += 1
    Δt = time() - t₀
    solved = max(dual, primal, compl) < ϵ
    tired = sum_counters(nlp) ≥ max_eval > 0 || Δt ≥ max_time > 0 || iter ≥ max_iter > 0
  end

  if solved
    status = :first_order
  elseif tired
    if Δt ≥ max_time > 0
      status = :max_time
    elseif sum_counters(nlp) ≥ max_eval > 0
      status = :max_eval
    elseif iter ≥ max_iter > 0
      status = :max_iter
    end
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution = x,
    objective = obj(nlp, x),
    dual_feas = dual,
    primal_feas = primal,
    elapsed_time = Δt,
    iter = iter,
  )
end
