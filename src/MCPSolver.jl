module MCPSolver

using SparseArrays: SparseArrays
using FastDifferentiation: FastDifferentiation as FD
using Symbolics: Symbolics
using LinearAlgebra: I, norm

using Infiltrator

include("SymbolicUtils.jl")
include("sparse_utils.jl")

include("mcp.jl")
include("solver.jl")

end # module MCPSolver
