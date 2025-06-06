export unconstrained_nlp

function unconstrained_nlp_set(; kwargs...)
  n = 30
  D = Diagonal([1 // 10 + 9 // 10 * (i - 1) // (n - 1) for i = 1:n])
  A = spdiagm(
    0 => 2 * ones(Rational{Int}, n),
    -1 => -ones(Rational{Int}, n - 1),
    1 => -ones(Rational{Int}, n - 1),
  )
  return [
    ADNLPModel(
      x -> (x[1] - 1)^2 + 4 * (x[2] - 1)^2,
      zeros(2),
      name = "(x₁ - 1)² + 4(x₂ - 1)²";
      kwargs...,
    ),
    ADNLPModel(x -> (x .- 1)' * D * (x .- 1), zeros(n), name = "Diagonal quadratic"; kwargs...),
    ADNLPModel(x -> (x .- 1)' * A * (x .- 1), zeros(n), name = "Tridiagonal quadratic"; kwargs...),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      name = "Rosenbrock";
      kwargs...,
    ),
    ADNLPModel(
      x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
      collect(1:n) ./ (n + 1),
      name = "Extended Rosenbrock";
      kwargs...,
    ),
  ]
end

"""
    unconstrained_nlp(solver; problem_set = unconstrained_nlp_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on unconstrained problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function unconstrained_nlp(solver; problem_set = unconstrained_nlp_set(), atol = 1e-6, rtol = 1e-6)
  @testset "Problem $(nlp.meta.name)" for nlp in problem_set
    stats = with_logger(NullLogger()) do
      solver(nlp)
    end
    ng0 = rtol != 0 ? norm(grad(nlp, nlp.meta.x0)) : 0
    ϵ = atol + rtol * ng0
    primal, dual = kkt_checker(nlp, stats.solution)
    @test all(abs.(dual) .< ϵ)
    @test all(abs.(primal) .< ϵ)
    @test stats.dual_feas < ϵ
    @test stats.status == :first_order
  end
end
