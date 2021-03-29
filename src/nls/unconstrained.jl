export unconstrained_nls

"""
    unconstrained_nls(solver)

Test the `solver` on unconstrained nonlinear least-squares problems.
"""
function unconstrained_nls(solver)
  n = 30
  D = Diagonal([0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n])
  A = spdiagm(0 => 2 * ones(n), -1 => -ones(n-1), -1 => -ones(n-1))
  @testset "Problem $(nls.meta.name)" for nls in [
    ADNLSModel(
      x -> [x[1] - 1; 2x[2] - 2],
      zeros(2),
      2,
      name = "(x₁ - 1)² + 4(x₂ - 1)²"
    ),
    ADNLSModel(
      x -> sqrt.(D) * (x .- 1),
      zeros(n),
      n,
      name = "Diagonal quadratic"
    ),
    ADNLSModel(
      x -> [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 4],
      zeros(2),
      2,
      name = "Non-zero residual"
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      name = "Rosenbrock"
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2); x[1] * x[2] - 1],
      [-1.2; 1.0],
      3,
      name = "Overdetermined"
    ),
    ADNLSModel(
      x -> [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]],
      (1:n) ./ (n + 1),
      2n - 2,
      name = "Extended Rosenbrock"
    ),
  ]
    stats = with_logger(NullLogger()) do
      solver(nls)
    end
    ng0 = norm(grad(nls, nls.meta.x0))
    @test isapprox(stats.solution, ones(nls.meta.nvar), atol=1e-6 * (ng0 + 1))
    @test stats.dual_feas < 1e-6 * (ng0 + 1)
    @test stats.status == :first_order
  end
end
