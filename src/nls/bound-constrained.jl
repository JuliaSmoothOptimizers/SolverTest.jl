export bound_constrained_nls

"""
    bound_constrained_nls(solver)

Test the `solver` on bound-constrained nonlinear least-squares problems.
"""
function bound_constrained_nls(solver)
  n = 30
  D = Diagonal([0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n])
  A = spdiagm(0 => 2 * ones(n), -1 => -ones(n-1), -1 => -ones(n-1))
  @testset "Problem $(nls.meta.name)" for nls in [
    ADNLSModel(
      x -> [x[1] - 1; 2x[2] - 2],
      zeros(2),
      2,
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Simple quadratic"
    ),
    ADNLSModel(
      x -> [x[1]^2 + x[2]^2 - 1; x[1] + x[2] - 4],
      zeros(2),
      2,
      zeros(2),
      2 * ones(2),
      name = "Non-zero residual"
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Rosenbrock inactive bounds"
    ),
    ADNLSModel(
      x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
      [-1.2; 1.0],
      2,
      [0.5; 0.25],
      [1.0; 1.5],
      name = "Rosenbrock active bounds"
    ),
    ADNLSModel(
      x -> [x[1] - 2; x[2] - 1],
      zeros(2),
      2,
      [1.0; 0.0],
      [1.0; 2.0],
      name = "One fixed variable"
    ),
    ADNLSModel(
      x -> x,
      zeros(n),
      n,
      ones(n),
      ones(n),
      name = "All variables fixed"
    ),
    ADNLSModel(
      x -> [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]],
      (1:n) ./ (n + 1),
      2n - 2,
      zeros(n),
      ones(n),
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
