"""
Utilities for constructing trajectory games, in which each player wishes to
solve a problem of the form:
                min_{τᵢ}   fᵢ(τ, θ)

where all vehicles must jointly satisfy the constraints
                           g̃(τ, θ) = 0
                           h̃(τ, θ) ≥ 0.

Here, τᵢ is the ith vehicle's trajectory, consisting of states and controls.
The shared constraints g̃ and h̃ incorporate dynamic feasibility, fixed initial
condition, actuator and state limits, environment boundaries, and
collision-avoidance.
"""

module TrajectoryExamples

push!(LOAD_PATH, pwd() * "/../src")
using MCPSolver:
    ParametricGame, OptimizationProblem, num_players, solve

using LazySets: LazySets
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    PolygonEnvironment,
    ProductDynamics,
    TimeSeparableTrajectoryGameCost,
    TrajectoryGame,
    GeneralSumCostStructure,
    num_players,
    time_invariant_linear_dynamics,
    unstack_trajectory,
    stack_trajectories,
    state_dim,
    control_dim,
    state_bounds,
    control_bounds,
    OpenLoopStrategy,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray, Block
using GLMakie: GLMakie
using Makie: Makie
using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter

include("utils.jl")
export build_parametric_game,
    WarmStartRecedingHorizonStrategy,
    pack_observations,
    unpack_observations,
    pack_parameters,
    unpack_trajectory,
    pack_parameters,
    unpack_parameters,
    parameter_mask,
    generate_initial_guess

include("lane_change.jl")

end # module TrajectoryExamples
