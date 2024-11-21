""" Test for the following QP:
                           min_x 0.5 xᵀ M x
                           s.t.  Ax - b ≥ 0.
Taking `y ≥ 0` as a Lagrange multiplier, we obtain the KKT conditions:
                             G(x, y) = Mx - Aᵀy = 0
                             0 ≤ y ⟂ H(x, y) = Ax - b ≥ 0.
"""

using Test: @testset, @test

using MCPSolver
using BlockArrays: BlockArray, Block, mortar, blocks

@testset "QPTestProblem" begin
    M = [2 1; 1 2]
    A = [1 0; 0 1]
    b = [1; 1]
    θ = zeros(2)

    G(x, y; θ) = M * x - A' * y - θ
    H(x, y; θ) = A * x - b
    K(z; θ) = begin
        x = z[1:size(M, 1)]
        y = z[(size(M, 1) + 1):end]

        [G(x, y; θ); H(x, y; θ)]
    end

    function check_solution(sol)
        @test all(abs.(G(sol.x, sol.y; θ)) .≤ 1e-3)
        @test all(H(sol.x, sol.y; θ) .≥ 0)
        @test all(sol.y .≥ 0)
        @test sum(sol.y .* H(sol.x, sol.y; θ)) ≤ 1e-3
        @test all(sol.s .≤ 1e-3)
        @test sol.kkt_error ≤ 1e-3
    end

    @testset "BasicCallableConstructor" begin
        mcp = MCPSolver.PrimalDualMCP(G, H, size(M, 1), length(b), size(M, 1))
        sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp; θ)

        check_solution(sol)
    end

    @testset "AlternativeCallableConstructor" begin
       mcp = MCPSolver.PrimalDualMCP(
            K,
            [fill(-Inf, size(M, 1)); fill(0, length(b))],
            fill(Inf, size(M, 1) + length(b)),
            size(M, 1)
        )
        sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp; θ)

        check_solution(sol)
    end
end

@testset "ParametricGameTests" begin
    lim = 0.5

    game = MCPSolver.ParametricGame(;
        test_point = mortar([[1, 1], [1, 1]]),
        test_parameter = mortar([[1, 1], [1, 1]]),
        problems = [
            MCPSolver.OptimizationProblem(;
                objective = (x, θi) -> sum((x[Block(1)] - θi) .^ 2),
                private_inequality = (xi, θi) -> -abs.(xi) .+ lim,
            ),
            MCPSolver.OptimizationProblem(;
                objective = (x, θi) -> sum((x[Block(2)] - θi) .^ 2),
                private_inequality = (xi, θi) -> -abs.(xi) .+ lim,
            ),
        ],
    )


    θ = mortar([[-1, 0], [1, 1]])
    (; primals, variables, kkt_error) = MCPSolver.solve(
        game,
        θ;
        tol = 1e-4,
    )

    @test isapprox.(
        primals[1],
        [max(-lim, θ[Block(1)][1]), min(lim, θ[Block(1)][2])],
        atol = tol,
    )
    @test isapprox.(
        primals[2],
        [max(-lim, θ[Block(2)][1]), min(lim, θ[Block(2)][2])],
        atol = tol,
    )
end
