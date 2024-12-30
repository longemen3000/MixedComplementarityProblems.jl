# Solver Benchmarks

Benchmarking `MixedComplementarityProblems` solver(s) against PATH.

## Instructions

Currently, this directory provides code to benchmark the `InteriorPoint` solver against `PATH`, accessed via `ParametricMCPs` and `PATHSolver`. The benchmark is a set of randomly-generated sparse quadratic programs with user-specified numbers of primal variables and inequality constraints. To run (with the REPL activated within this directory):

```julia
julia> Revise.includet("SolverBenchmarks.jl")
julia> data = SolverBenchmarks.benchmark()
julia> SolverBenchmarks.summary_statistics(data)
```