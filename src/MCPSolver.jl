module MCPSolver

using SparseArrays: SparseArrays
using FastDifferentiation: FastDifferentiation as FD
using Symbolics: Symbolics
using LinearAlgebra: I, norm, eigvals
using BlockArrays: blocks, blocksizes
using TrajectoryGamesBase: to_blockvector

include("SymbolicUtils.jl")
include("mcp.jl")
include("solver.jl")
include("game.jl")
include("AutoDiff.jl")

end # module MCPSolver
