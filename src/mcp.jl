""" Store key elements of the primal-dual KKT system for a MCP composed of
functions G(.) and H(.) such that
                             0 = G(x, y)
                             0 ≤ H(x, y) ⟂ y ≥ 0.

The primal-dual system arises when we introduce slack variable `s` and set
                             G(x, y)     = 0
                             H(x, y) - s = 0
                             s ⦿ y - ϵ   = 0
for some ϵ > 0. Define the function `F(z; ϵ)` to return the left hand side of this
system of equations, where `z = [x; y; s]`.
"""
struct PrimalDualMCP{T1,T2}
    "A callable `F(x, y, s; ϵ)` which computes the KKT error in the primal-dual system."
    F::T1
    "A callable `∇F(x, y, s; ϵ)` which stores the Jacobian of the KKT error wrt z."
    ∇F::T2
    "Dimension of unconstrained variable."
    unconstrained_dimension::Int
    "Dimension of constrained variable."
    constrained_dimension::Int
end

"Helper to construct a PrimalDualMCP from callable functions `G(.)` and `H(.)`."
function to_symbolic_mcp(
    G,
    H,
    unconstrained_dimension,
    constrained_dimension;
    backend = SymbolicUtils.SymbolicsBackend(),
    backend_options = (;)
)
    x_symbolic = SymbolicUtils.make_variables(backend, :x, unconstrained_dimension)
    y_symbolic = SymbolicUtils.make_variables(backend, :y, constrained_dimension)
    s_symbolic = SymbolicUtils.make_variables(backend, :s, constrained_dimension)
    G_symbolic = G(x_symbolic, y_symbolic)
    H_symbolic = H(x_symbolic, y_symbolic)

    PrimalDualMCP(
        G_symbolic, H_symbolic, x_symbolic, y_symbolic; backend, backend_options)
end

"Construct a PrimalDualMCP from symbolic expressions of G(.) and H(.)."
function PrimalDualMCP(
    G_symbolic::Vector{T},
    H_symbolic::Vector{T},
    x_symbolic::Vector{T},
    y_symbolic::Vector{T};
    backend = SymbolicUtils.SymbolicsBackend(),
    backend_options = (;)
) where {T<:Union{FD.Node,Symbolics.Num}}
    # Create symbolic slack variable `s` and parameter `ϵ`.
    s_symbolic = SymbolicUtils.make_variables(backend, :s, length(y_symbolic))
    ϵ_symbolic = only(SymbolicUtils.make_variables(backend, :ϵ, 1))
    z_symbolic = [x_symbolic; y_symbolic; s_symbolic]

    F_symbolic = [
        G_symbolic;
        H_symbolic - s_symbolic;
        s_symbolic .* y_symbolic .- ϵ_symbolic
    ]

    F = let
        _F = SymbolicUtils.build_function(
            F_symbolic,
            [z_symbolic; ϵ_symbolic];
            in_place = false,
            backend_options,
        )

        (x, y, s; ϵ) -> _F([x; y; s; ϵ])
    end

    ∇F = let
        ∇F_symbolic = SymbolicUtils.sparse_jacobian(F_symbolic, z_symbolic)
        _∇F = SymbolicUtils.build_function(
            ∇F_symbolic,
            [z_symbolic; ϵ_symbolic];
            in_place = false,
            backend_options,
        )

        (x, y, s; ϵ) -> _∇F([x; y; s; ϵ])
    end

    PrimalDualMCP(F, ∇F, length(x_symbolic), length(y_symbolic))
end

"""Construct a PrimalDualMCP from K(z) ⟂ z̲ ≤ z ≤ z̅.
NOTE: Assumes that all upper bounds are Inf, and lower bounds are either
-Inf or 0.
"""
function PrimalDualMCP(
    K_symbolic::Vector{T},
    z_symbolic::Vector{T},
    lower_bounds::Vector,
    upper_bounds::Vector;
    backend_options = (;)
    ) where {T<:Union{FD.Node,Symbolics.Num}}
    assert(
        all(isinf.(lower_bounds)) &&
        all(isinf.(upper_bounds) .|| upper_bounds .== 0)
    )

    unconstrained_indices = findall(isinf, upper_bounds)
    constrained_indices = findall(!isinf, upper_bounds)

    G_symbolic = K_symbolic[unconstrained_indices]
    H_symbolic = K_symbolic[constrained_indices]
    x_symbolic = z_symbolic[unconstrained_indices]
    y_symbolic = z_symbolic[constrained_indices]

    PrimalDualMCP(G_symbolic, H_symbolic, x_symbolic, y_symbolic; backend_options)
end
