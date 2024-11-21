module MCPSolver

using SparseArrays: SparseArrays
using FastDifferentiation: FastDifferentiation as FD
using Symbolics: Symbolics
using LinearAlgebra: I, norm, eigvals
using BlockArrays: blocks, blocksizes
using TrajectoryGamesBase: to_blockvector

using Infiltrator

include("SymbolicUtils.jl")
include("mcp.jl")
include("solver.jl")
include("game.jl")

end # module MCPSolver
