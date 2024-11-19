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
using LevelTwoInverseGames:
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
using PATHSolver: PATHSolver
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

"Zygote-friendly trajectory unpacking utility."
function unpack_trajectory_zygote(flat_trajectories; dynamics::ProductDynamics)
    horizon = Int(
        length(flat_trajectories[1]) /
        (state_dim(dynamics, 1) + control_dim(dynamics, 1)),
    )

    map(1:num_players(dynamics), flat_trajectories) do ii, traj
        num_states = state_dim(dynamics, ii) * horizon
        X = reshape(traj[1:num_states], (state_dim(dynamics, ii), horizon))
        U = reshape(
            traj[(num_states + 1):end],
            (control_dim(dynamics, ii), horizon),
        )

        (; xs = eachcol(X) |> collect, us = eachcol(U) |> collect)
    end
end

export unpack_trajectory_zygote

include("lane_change.jl")

end # module TrajectoryExamples
