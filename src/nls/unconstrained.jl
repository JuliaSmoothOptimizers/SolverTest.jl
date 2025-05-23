export unconstrained_nls

function unconstrained_nls_set(; kwargs...)
  n = 30
  D = Diagonal([1 // 10 + 9 // 10 * (i - 1) // (n - 1) for i = 1:n])
  A = spdiagm(
    0 => 2 * ones(Rational{Int}, n),
    -1 => -ones(Rational{Int}, n - 1),
    1 => -ones(Rational{Int}, n - 1),
  )
  return [
    ADNLSModel(x -> [x[1] - 1; 2x[2] - 2], zeros(2), 2, name = "(x₁ - 1)² + 4(x₂ - 1)²"; kwargs...),
    ADNLSModel(x -> sqrt.(D) * (x .- 1), zeros(n), n, name = "Diagonal quadratic"; kwargs...),
    ADNLSModel(
      x -> [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 4],
      zeros(2),
      2,
      name = "Non-zero residual";
      kwargs...,
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      name = "Rosenbrock";
      kwargs...,
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2); x[1] * x[2] - 1],
      [-1.2; 1.0],
      3,
      name = "Overdetermined";
      kwargs...,
    ),
    ADNLSModel(
      x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(n - 1)]; [x[i] - 1 for i = 1:(n - 1)]],
      collect(1:n) ./ (n + 1),
      2n - 2,
      name = "Extended Rosenbrock";
      kwargs...,
    ),
  ]
end

"""
    unconstrained_nls(solver; problem_set = unconstrained_nls_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on unconstrained nonlinear least-squares problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function unconstrained_nls(solver; problem_set = unconstrained_nls_set(), atol = 1e-6, rtol = 1e-6)
  @testset "Problem $(nls.meta.name)" for nls in problem_set
    stats = with_logger(NullLogger()) do
      solver(nls)
    end
    ng0 = rtol != 0 ? norm(grad(nls, nls.meta.x0)) : 0
    ϵ = atol + rtol * ng0
    primal, dual = kkt_checker(nls, stats.solution)
    @test all(abs.(dual) .< ϵ)
    @test primal == [] || all(abs.(primal) .< ϵ)
    @test stats.dual_feas < ϵ
    @test stats.status == :first_order
  end
end
