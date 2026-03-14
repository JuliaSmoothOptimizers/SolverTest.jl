export inequality_constrained_nlp

function inequality_constrained_nlp_set()
  return [
    ADNLPModel( # ones(2),
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2],
      [1.0; 2.0],
      x -> [4x[1] + 6x[2] - 10],
      [-Inf],
      [0.0],
      name = "Simple quadratic problem with a linear constraint",
    ),
    ADNLPModel( # [0.990099; 0.980296],
      x -> (x[1] - 1)^2 + 0.01 * x[2],
      [-1.2; 1.0],
      x -> [10 * (x[2] - x[1]^2)],
      [0.0],
      [Inf],
      name = "Simple quadratic problem with a quadratic constraint",
    ),
    ADNLPModel(
      x -> x[1] - x[2],
      [-10.0; 10.0],
      x -> [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1],
      [0.0],
      [Inf],
      name = "HS10",
    ),
    ADNLPModel( # ones(2),
      x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      x -> [x[1]^2 + x[2]^2],
      [0.0],
      [4.0],
      name = "Rosenbrock with circle constraint",
    ),
    ADNLPModel( # [5.196152; 1.732051],
      x -> -x[1],
      [2.0; 1.0],
      x -> [x[1]^2 + x[2]^2; x[1] * x[2]],
      [25.0; 9.0],
      [30.0; 12.0],
      name = "Unbounded objective with inequalities",
    ),
    ADNLPModel(
      x -> (x[1] - 2)^2 + (x[2] - 1)^2,
      [-10.0; 10.0],
      x -> [x[1] + x[2], -x[1]^2 + x[2]],
      [-Inf; 0.0],
      [2; Inf],
      name = "HS22",
    ),
  ]
end

"""
    inequality_constrained_nlp(solver; problem_set = inequality_constrained_nlp_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on inequality-constrained problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function inequality_constrained_nlp(
  solver;
  problem_set = inequality_constrained_nlp_set(),
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
    @test stats.primal_feas < ϵ
    @test stats.status == :first_order
  end
end
