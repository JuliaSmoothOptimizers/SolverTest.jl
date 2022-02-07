export equality_constrained_nls

function equality_constrained_nls_set()
  n = 10
  return [
    ADNLSModel(
      x -> [x[1] - 1],
      [-1.2; 1.0],
      1,
      x -> [10 * (x[2] - x[1]^2)],
      zeros(1),
      zeros(1),
      name = "HS6",
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      x -> [(x[1] - 2)^2 + (x[2] - 2)^2 - 2],
      zeros(1),
      zeros(1),
      name = "Rosenbrock with (x₁-2)²+(x₂-2)²=2",
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      x -> [sum(x) - 2],
      zeros(1),
      zeros(1),
      name = "Rosenbrock with ∑x = 2",
    ),
    ADNLSModel(
      x -> [x[1] - 1; x[2] - 1],
      -ones(2),
      2,
      x -> [sum(x) - 2],
      zeros(1),
      zeros(1),
      name = "linear residual and linear constraints",
    ),
    ADNLSModel(
      x -> [x[1] - 1; x[2] - 1],
      -ones(2),
      2,
      x -> [sum(x .^ 2) - 2; x[2] - x[1]^2],
      zeros(2),
      zeros(2),
      name = "linear residual and quad constraints",
    ),
    ADNLSModel(
      x -> [x[1] - x[i] for i = 2:n],
      [1.0j for j = 1:n] / n,
      n - 1,
      x -> [sum(x) - n],
      zeros(1),
      zeros(1),
      name = "F_under and linear constraints",
    ),
    ADNLSModel(
      x -> [x[1] - x[i] for i = 2:n],
      [1.0j for j = 1:n] / n,
      n - 1,
      x -> [sum(x .^ 2) - n; prod(x) - 1],
      zeros(2),
      zeros(2),
      name = "F_under and quad constraints",
    ),
    ADNLSModel(
      x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(n - 1)]; [x[i] - 1 for i = 1:(n - 1)]],
      0.9 * ones(n),
      2(n - 1),
      x -> [sum(x) - n],
      zeros(1),
      zeros(1),
      name = "F_larger and linear constraints",
    ),
    ADNLSModel(
      x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(n - 1)]; [x[i] - 1 for i = 1:(n - 1)]],
      0.9 * ones(n),
      2(n - 1),
      x -> [sum(x .^ 2) - n; prod(x) - 1],
      zeros(2),
      zeros(2),
      name = "F_larger and quad constraints",
    ),
  ]
end
"""
    equality_constrained_nls(solver; problem_set = equality_constrained_nls_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on equality-constrained problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function equality_constrained_nls(
  solver;
  problem_set = equality_constrained_nls_set(),
  atol = 1e-6,
  rtol = 1e-6,
)
  @testset "Problem $(nls.meta.name)" for nls in problem_set
    stats = with_logger(NullLogger()) do
      solver(nls)
    end
    ng0 = rtol != 0 ? norm(grad(nls, nls.meta.x0)) : 0
    ϵ = atol + rtol * ng0
    primal, dual = kkt_checker(nls, stats.solution, feas_tol = atol, bound_tol = atol)
    @test all(dual .< ϵ)
    @test primal == [] || all(primal .< ϵ)
    @test stats.dual_feas < ϵ
    @test stats.primal_feas < ϵ
    @test stats.status == :first_order
  end
end
