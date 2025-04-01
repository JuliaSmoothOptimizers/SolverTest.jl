export bound_constrained_nlp

function bound_constrained_nlp_set(; kwargs...)
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
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Simple quadratic";
      kwargs...,
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Rosenbrock inactive bounds";
      kwargs...,
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      [0.5; 0.25],
      [1.0; 1.5],
      name = "Rosenbrock active bounds";
      kwargs...,
    ),
    ADNLPModel(
      x -> (x[1] - 2)^2 + (x[2] - 1)^2 - 1,
      zeros(2),
      [1.0; 0.0],
      [1.0; 2.0],
      name = "One fixed variable";
      kwargs...,
    ),
    ADNLPModel(
      x -> sum(x .^ 2) - n,
      zeros(n),
      ones(n),
      ones(n),
      name = "All variables fixed";
      kwargs...,
    ),
    ADNLPModel(
      x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
      collect(1:n) ./ (n + 1),
      zeros(n),
      ones(n),
      name = "Extended Rosenbrock";
      kwargs...,
    ),
  ]
end

"""
    bound_constrained_nlp(solver; problem_set = bound_constrained_nlp_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on bound-constrained problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function bound_constrained_nlp(
  solver;
  problem_set = bound_constrained_nlp_set(),
  atol = 1e-6,
  rtol = 1e-6,
)
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
