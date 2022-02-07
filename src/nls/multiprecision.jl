export multiprecision_nls

"""
    multiprecision_nls(solver, problem_type; precisions=[Float16, …, BigFloat])

Test that `solver` solves a problem of type `problem_type` on various `precisions`.
The `problem_type` can be
- :unc
- :bnd
- :equ
- :ineq
- :eqnbnd
- :gen
"""
function multiprecision_nls(solver, ptype; precisions = (Float16, Float32, Float64, BigFloat))
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  c(x) = [x[1]^2 + x[2]^2]
  c2(x) = [c(x); x[2] - x[1]^2 / 10]
  @testset "Precision $T for ptype $ptype" for T in precisions
    x0 = T[-1.2; 1.0]
    ℓ, u = zeros(T, 2), 2 * ones(T, 2)
    nls = if ptype == :unc
      ADNLSModel(F, x0, 2)
    elseif ptype == :bnd
      ADNLSModel(F, x0, 2, ℓ, u)
    elseif ptype == :equ
      ADNLSModel(F, x0, 2, c, T[2.0], T[2.0])
    elseif ptype == :ineq
      ADNLSModel(F, x0, 2, c, T[0.0], T[2.0])
    elseif ptype == :eqnbnd
      ADNLSModel(F, x0, 2, ℓ, u, c, T[2.0], T[2.0])
    elseif ptype == :gen
      ADNLSModel(F, x0, 2, ℓ, u, c2, T[2.0; 0.0], T[2.0; Inf])
    else
      error("Unexpected ptype $ptype")
    end
    ϵ = eps(T)^T(1 / 4)

    ng0 = norm(grad(nls, nls.meta.x0))

    stats = with_logger(NullLogger()) do
      solver(nls, atol = ϵ, rtol = ϵ)
    end
    @test eltype(stats.solution) == T
    @test stats.objective isa T
    @test stats.dual_feas isa T
    @test stats.primal_feas isa T
    primal, dual = kkt_checker(nls, stats.solution, feas_tol = ϵ, bound_tol = ϵ)
    @test all(dual .< ϵ * ng0 + ϵ)
    @test primal == [] || all(primal .< ϵ * ng0 + ϵ)
    @test stats.dual_feas < ϵ * ng0 + ϵ
  end
end

# @testset "Multiprecision" begin
#   for T in (Float16, Float32, Float64, BigFloat)
#     nls = ADNLSModel(x -> [x[1] - 1; x[2] - x[1]^2], T[-1.2; 1.0], 2)
#     ϵ = eps(T)^T(1/4)

#     g0 = grad(nls, nls.meta.x0)
#     ng0 = norm(g0)

#     stats = with_logger(NullLogger()) do
#       solver(nls, max_eval=-1, atol=ϵ, rtol=ϵ)
#     end
#     @test eltype(stats.solution) == T
#     @test stats.objective isa T
#     @test stats.dual_feas isa T
#     @test stats.primal_feas isa T
#     @test isapprox(stats.solution, ones(T, 2), atol=ϵ * ng0 * 10)
#     @test isapprox(stats.objective, zero(T), atol=ϵ * ng0)
#     @test stats.dual_feas < ϵ * ng0 + ϵ
#   end
# end
