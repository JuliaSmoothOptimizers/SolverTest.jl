export multiprecision_nlp

"""
    multiprecision_nlp(solver, problem_type; precisions=[Float16, …, BigFloat])

Test that `solver` solves a problem of type `problem_type` on various `precisions`.
The `problem_type` can be
- :unc
- :bnd
- :equ
- :ineq
- :eqnbnd
- :gen
"""
function multiprecision_nlp(Solver, ptype)
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  c(x) = [x[1]^2 + x[2]^2]
  c2(x) = [c(x); x[2] - x[1]^2 / 10]
  @testset "Precision $T for ptype $ptype" for T in (Float16, Float32, Float64, BigFloat)
    x0 = T[-1.2; 1.0]
    ℓ, u = zeros(T, 2), 2 * ones(T, 2)
    nlp = if ptype == :unc
      ADNLPModel(f, x0)
    elseif ptype == :bnd
      ADNLPModel(f, x0, ℓ, u)
    elseif ptype == :equ
      ADNLPModel(f, x0, c, [2.0], [2.0])
    elseif ptype == :ineq
      ADNLPModel(f, x0, c, [-1.0], [1.0])
    elseif ptype == :eqnbnd
      ADNLPModel(f, x0, ℓ, u, c, [2.0], [2.0])
    elseif ptype == :gen
      ADNLPModel(f, x0, ℓ, u, c2, [2.0; 0.0], [2.0; Inf])
    else
      error("Unexpected ptype $ptype")
    end
    ϵ = eps(T)^T(1/4)

    ng0 = norm(grad(nlp, nlp.meta.x0))

    solver = Solver(nlp.meta)
    stats = with_logger(NullLogger()) do
      solve!(solver, nlp, atol=ϵ, rtol=ϵ)
    end
    @test eltype(stats.solution) == T
    @test stats.objective isa T
    @test stats.dual_feas isa T
    @test stats.primal_feas isa T
    @test isapprox(stats.solution, ones(T, 2), atol=ϵ * ng0 * 10)
    @test stats.dual_feas < ϵ * ng0 + ϵ
  end
end
