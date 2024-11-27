# MCPSolver.jl

[![CI](https://github.com/CLeARoboticsLab/MCPSolver.jl/actions/workflows/test.yml/badge.svg)](https://github.com/CLeARoboticsLab/MCPSolver.jl/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD-new)](https://opensource.org/license/bsd-3-clause)

This package provides an easily-customizable interface for expressing mixed complementarity problems (MCPs) which are defined in terms of an arbitrary vector of parameters. `MCPSolver` implements a reasonably high-performance interior point method for solving these problems, and integrates with `ChainRulesCore` and `ForwardDiff` to enable automatic differentiation of solutions with respect to problem parameters.

## What are MCPs?

Mixed complementarity problems (MCPs) are a class of mathematical program, and they arise in a wide variety of application problems. In particular, one way they can arise is via the KKT conditions of nonlinear programs and noncooperative games. This package provides a utility for constructing MCPs from (parameterized) games, cf. `src/game.jl` for further details. To see the connection between KKT conditions and MCPs, read the next section.

## Why this package?

As discussed below, this package replicates functionality already available in [ParametricMCPs](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl). Our intention here is to provide an easily customizable and open-source solver with efficiency and reliability that is at least comparable with the [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver which `ParametricMCPs` uses under the hood (actually, it hooks into the interface to the `PATH` binaries which is provided by another wonderful package, [PATHSolver](https://github.com/chkwon/PATHSolver.jl)). Hopefully, users will find it useful to modify the interior point solver provided in this package for their own application problems, use it for highly parallelized implementations (since it is in pure Julia), etc.

## Quickstart guide

Suppose we have the following quadratic program:
```math
&\min_x~ &\frac{1}{2} x^\top M x - θ^\top x\\
&s.t. &Ax - b \ge 0.
```

The KKT conditions for this problem can be expressed as follows:
```math
G(x, y; θ) = M x - θ - A^\top y &= 0\\
H(x, y; θ) = A x - b &\ge 0\\
y &\ge 0\\
y^\top H(x, y; θ) = 0,
```
where $y$ is the Lagrange multiplier associated to the constraint $Ax - b \ge 0$ in the original problem.

This is precisely a MCP, whose standard form is:
```math
G(x, y; θ) &= 0\\
0 \le y \perp H(x, y; θ) &\ge 0.
```

Now, we can encode this problem and solve it using `MCPSolver` as follows:

```julia
using MCPSolver

M = [2 1; 1 2]
A = [1 0; 0 1]
b = [1; 1]
θ = rand(2)

G(x, y; θ) = M * x - θ - A' * y
H(x, y; θ) = A * x - b

mcp = MCPSolver.PrimalDualMCP(
    G,
    H;
    unconstrained_dimension = size(M, 1),
    constrained_dimension = length(b),
    parameter_dimension = size(M, 1),
)
sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp, θ)
```

The solver can easily be warm-started from a given initial guess:
```julia
sol = MCPSolver.solve(
    MCPSolver.InteriorPoint(),
    mcp,
    θ;
    x₀ = # your initial guess
    y₀ = # your **positive** initial guess
)
```

Note that the initial guess for the $y$ variable must be elementwise positive. This is because we are using an interior point method; for further details, refer to `src/solver.jl`.

Finally, `MCPSolver` integrates with `ChainRulesCore` and `ForwardDiff` so you can differentiate through the solver itself! For example, suppose we wanted to find the value of $\theta$ in the problem above which solves
```math
&\min_{\theta, x, y}~ &\overbrace{\|x\|_2^2 + \|y\|_2^2}^{f(x, y)}\\
&s.t. &(x, y) solves \text{MCP}(\theta).
```
We could do so by initializing with a particular value of $\theta$ and then iteratively descending the gradient $\nabla_\theta f$, which we can easily compute via:
```julia
mcp = MCPSolver.PrimalDualMCP(
    G,
    H;
    unconstrained_dimension = size(M, 1),
    constrained_dimension = length(b),
    parameter_dimension = size(M, 1),
    compute_sensitivities = true,
)

function f(θ)
    sol = MCPSolver.solve(MCPSolver.InteriorPoint(), mcp, θ)
    sum(sol.x .^ 2) + sum(sol.y .^ 2)
end

∇f = only(Zygote.gradient(f, θ))
```

## Acknowledgement and future plans

This project inherits many key ideas from [ParametricMCPs](https://github.com/JuliaGameTheoreticPlanning/ParametricMCPs.jl), which provides essentially identical functionality but which currently only supports the (closed-source, but otherwise excellent [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver). Ultimately, this `MCPSolver` will likely merge with `ParametricMCPs` to provide an identical frontend and allow users a flexible choice of backend solver. Currently, `MCPSolver` replicates a substantially similar interface as that provided by `ParametricMCPs`, but there are some (potentially annoying) differences that users should take care to notice, e.g., in the function signature for `solve(...)`.

## Other related projects

If you are curious about other related software, consider checking out [JuliaGameTheoreticPlanning](https://github.com/orgs/JuliaGameTheoreticPlanning/repositories).
