module MixedComplementarityProblems

using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using BlockArrays: blocks, blocksizes, BlockArray
using SymbolicTracingUtils: SymbolicTracingUtils as SymbolicTracingUtils

to_blockvector(block_dimensions) = Base.Fix2(BlockArray,block_dimensions)

include("mcp.jl")
include("solver.jl")
include("game.jl")
include("AutoDiff.jl")

export PrimalDualMCP, solve, ParametricGame, OptimizationProblem

end # module MixedComplementarityProblems
