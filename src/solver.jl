abstract type SolverType end
struct InteriorPoint <: SolverType end

""" Basic interior point solver, based on Nocedal & Wright, ch. 19.
Computes step directions `δz` by solving the relaxed primal-dual system, i.e.
                         ∇F(z; ϵ) δz = -F(z; ϵ).

Given a step direction `δz`, performs a "fraction to the boundary" linesearch,
i.e., for `(x, s)` it chooses step size `α_s` such that
              α_s = max(α ∈ [0, 1] : s + δs ≥ (1 - τ) s)
and for `y` it chooses step size `α_s` such that
              α_y = max(α ∈ [0, 1] : y + δy ≥ (1 - τ) y).

A typical value of τ is 0.995. Once we converge to ||F(z; \epsilon)|| ≤ ϵ,
we typically decrease ϵ by a factor of 0.1 or 0.2, with smaller values chosen
when the previous subproblem is solved in fewer iterations.
"""
function solve(::InteriorPoint, mcp::PrimalDualMCP, x₀, y₀; tol = 1e-4)
    x = x₀
    y = y₀
    s = ones(length(y))

    ϵ = 1.0
    kkt_error = Inf
    while kkt_error > tol
        iters = 1
        while kkt_error > ϵ
            F = mcp.F(x, y, s, ϵ)
            δz = -mcp.∇F(x, y, s, ϵ) \ F

            # TODO! Linesearch...
            # TODO! Update variables...

            kkt_error = norm(F)
            iters += 1
        end

        ϵ *= 1 - exp(-iters)
    end

    (; x, y, s, kkt_error)
end
