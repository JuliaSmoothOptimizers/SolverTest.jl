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
function multiprecision_nlp(solver, ptype; precisions = (Float16, Float32, Float64, BigFloat))
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  c(x) = [x[1]^2 + x[2]^2]
  c2(x) = [c(x); x[2] - x[1]^2 / 10]
  @testset "Precision $T for ptype $ptype" for T in precisions
    x0 = T[-1.2; 1.0]
    ℓ, u = zeros(T, 2), 2 * ones(T, 2)
    nlp = if ptype == :unc
      ADNLPModel(f, x0)
    elseif ptype == :bnd
      ADNLPModel(f, x0, ℓ, u)
    elseif ptype == :equ
      ADNLPModel(f, x0, c, T[2.0], T[2.0])
    elseif ptype == :ineq
      ADNLPModel(f, x0, c, T[0.0], T[2.0])
    elseif ptype == :eqnbnd
      ADNLPModel(f, x0, ℓ, u, c, T[2.0], T[2.0])
    elseif ptype == :gen
      ADNLPModel(f, x0, ℓ, u, c2, T[2.0; 0.0], T[2.0; Inf])
    else
      error("Unexpected ptype $ptype")
    end
    ϵ = eps(T)^T(1 / 4)

    ng0 = norm(grad(nlp, nlp.meta.x0))

    stats = with_logger(NullLogger()) do
      solver(nlp, atol = ϵ, rtol = ϵ)
    end
    @test eltype(stats.solution) == T
    @test stats.objective isa T
    @test stats.dual_feas isa T
    @test stats.primal_feas isa T
    primal, dual = kkt_checker(nlp, stats.solution, feas_tol = ϵ, bound_tol = ϵ)
    @test all(dual .< ϵ * ng0 + ϵ)
    @test primal == [] || all(primal .< ϵ * ng0 + ϵ)
    @test stats.dual_feas < ϵ * ng0 + ϵ
  end
end
