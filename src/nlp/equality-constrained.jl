export equality_constrained_nlp

function equality_constrained_nlp_set()
  n = 30
  return [
    ADNLPModel(
      x -> 2x[1]^2 + x[1] * x[2] + x[2]^2 - 9x[1] - 9x[2] + 14,
      [1.0; 2.0],
      x -> [4x[1] + 6x[2] - 10],
      zeros(1),
      zeros(1),
      name = "Simple quadratic problem",
    ),
    ADNLPModel(
      x -> (x[1] - 1)^2,
      [-1.2; 1.0],
      x -> [10 * (x[2] - x[1]^2)],
      zeros(1),
      zeros(1),
      name = "HS6",
    ),
    ADNLPModel(
      x -> -x[1] + 1,
      [0.5; 1 / 3],
      x -> [
        16x[1]^2 + 9x[2]^2 - 25
        4x[1] * 3x[2] - 12
      ],
      zeros(2),
      zeros(2),
      name = "scaled HS8",
    ),
    ADNLPModel(
      x -> dot(x, x) - n,
      zeros(n),
      x -> [sum(x) - n],
      zeros(1),
      zeros(1),
      name = "‖x‖² s.t. ∑x = n",
    ),
    ADNLPModel(
      x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
      [-1.2; 1.0],
      x -> [sum(x) - 2],
      [0.0],
      [0.0],
      name = "Rosenbrock with ∑x = 2",
    ),
  ]
end

"""
    equality_constrained_nlp(solver; problem_set = equality_constrained_nlp_set(), atol = 1e-6, rtol = 1e-6)

Test the `solver` on equality-constrained problems.
If `rtol` is non-zero, the relative error uses the gradient at the initial guess.
"""
function equality_constrained_nlp(
  solver;
  problem_set = equality_constrained_nlp_set(),
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
