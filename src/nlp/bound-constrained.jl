export bound_constrained_nlp

"""
    bound_constrained_nlp(solver)

Test the `solver` on bound-constrained problems.
"""
function bound_constrained_nlp(solver)
  n = 30
  D = Diagonal([0.1 + 0.9 * (i - 1) / (n - 1) for i = 1:n])
  A = spdiagm(0 => 2 * ones(n), -1 => -ones(n-1), -1 => -ones(n-1))
  @testset "Problem $(nlp.meta.name)" for nlp in [
    ADNLPModel(
      x -> (x[1] - 1)^2 + 4 * (x[2] - 1)^2,
      zeros(2),
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Simple quadratic"
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      [0.5; 0.25],
      [1.2; 1.5],
      name = "Rosenbrock inactive bounds"
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      [0.5; 0.25],
      [1.0; 1.5],
      name = "Rosenbrock active bounds"
    ),
    ADNLPModel(
      x -> (x[1] - 2)^2 + (x[2] - 1)^2 - 1,
      zeros(2),
      [1.0; 0.0],
      [1.0; 2.0],
      name = "One fixed variable"
    ),
    ADNLPModel(
      x -> sum(x .^ 2) - n,
      zeros(n),
      ones(n),
      ones(n),
      name = "All variables fixed"
    ),
    ADNLPModel(
      x -> sum(100 * (x[i+1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:n-1),
      (1:n) ./ (n + 1),
      zeros(n),
      ones(n),
      name = "Extended Rosenbrock"
    ),
  ]
    stats = with_logger(NullLogger()) do
      solver(nlp)
    end
    ng0 = norm(grad(nlp, nlp.meta.x0))
    @test isapprox(stats.solution, ones(nlp.meta.nvar), atol=1e-6 * (ng0 + 1))
    @test isapprox(stats.objective, 0.0, atol=1e-6 * (ng0 + 1))
    @test stats.dual_feas < 1e-6 * (ng0 + 1)
    @test stats.status == :first_order
  end
end
