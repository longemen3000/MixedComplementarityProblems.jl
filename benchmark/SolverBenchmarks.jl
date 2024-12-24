"Module for benchmarking different solvers against one another."
module SolverBenchmarks

using MixedComplementarityProblems: MixedComplementarityProblems
using ParametricMCPs: ParametricMCPs
using Random: Random
using Statistics: Statistics
using PATHSolver: PATHSolver
using ProgressMeter: @showprogress

include("path.jl")

end # module SolverBenchmarks
