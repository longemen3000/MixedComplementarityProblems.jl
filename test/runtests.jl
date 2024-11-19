""" Test for the following QP:
                           min_x 0.5 xᵀ M x
                           s.t.  Ax - b ≥ 0.
Taking `y ≥ 0` as a Lagrange multiplier, we obtain the KKT conditions:
                             G(x, y) = Mx - Aᵀy = 0
                             0 ≤ y ⟂ H(x, y) = Ax - b ≥ 0.
"""

using Test: @testset, @test
using MCPSolver

@testset "QPTestProblem" begin
    M = [2 1; 1 2]
    A = [1 0; 0 1]
    b = [1; 1]

    G(x, y) = M * x - A' * y
    H(x, y) = A * x - b
    F(z) = begin
        x = z[1:size(M, 1)]
        y = z[(size(M, 1) + 1):end]

        [G(x, y); H(x, y)]
    end

    function check_solution(sol)
        @test all(abs.(G(sol.x, sol.y)) .≤ 1e-3)
        @test all(H(sol.x, sol.y) .≥ 0)
        @test all(sol.y .≥ 0)
        @test sum(sol.y .* H(sol.x, sol.y)) ≤ 1e-3
        @test all(sol.s .≤ 1e-3)
        @test sol.kkt_error ≤ 1e-3
    end

    @testset "BasicCallableConstructor" begin
        mcp = MCPSolver.to_symbolic_mcp(G, H, size(M, 1), length(b))
        sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp)

        check_solution(sol)
    end

    @testset "AlternativeCallableConstructor" begin
        mcp = MCPSolver.PrimalDualMCP(
            F,
            [fill(-Inf, size(M, 1)); fill(0, length(b))],
            fill(Inf, size(M, 1) + length(b)),
        )
        sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp)

        check_solution(sol)
    end
end
