#1. Important Includes:
#-------------------------------------------------------------------------------
println(workers())

@everywhere using SegwayRobot3# Include Environment
@everywhere using Knet
@everywhere using AutoGrad
@everywhere using PyCall

#using ModelVisualisations #To have videos
using ProgressMeter # To estimate the simulation time
using PyPlot        # To visualize the improvement
using JLD2 #To save the results

include("data_types.jl") #Required Data types
include("trajectory_generating.jl") #Trajectory and Episodes
include("net_functionality.jl") #Neuronal Network
include("parallel_execution.jl") #To run things in parallel
include("utils.jl") #Algo-Specifics
include("TRPO.jl") #Including 2nd order derivatives, line-seach and trust region
include("GAE.jl") #Include General Advantage Estimation
include("PPO.jl") #Include the Surrogate Objective calculation
include("visualization.jl") #Visualization
norm_active = true #set TRUE if normalization should be activated
include("layer_norm.jl") #Normalization


include("NIPS2018.jl") #If trained on the nips environment

#2: Initialize
#-------------------------------------------------------------------------------

#2.1: Environment Parameters
#------------------------
params           = Params()
params.dt        = 1/100;   #Timestep
params.t_horizon = 5;       #Time until end
params.l_traj    = length(0:params.dt:params.t_horizon); #Total timesteps per episode
params.u_range   = u_range    #Range of motor power (Scale of NN output)
params.s_init    = s_init; #Init_position: Pendulum on the bottom!
params.l_s_t     = length(params.s_init) #Contains number of relevant input states
params.l_out     = length(params.u_range) #Contains output dimension

#2.2: Parameters of the Network and Training
#------------------------
params.structure_net_pl  = Int[params.l_s_t + 0; 2^6; 2^6; params.l_out] #Net structure policy function
params.structure_net_vf  = Int[params.l_s_t + 0; 2^6; 2^6; 1] #Net structure value function
params.structure_net_qf  = Int[params.l_s_t + params.l_out; 2^8; 2^8; 1] #Net structure Q function 1
params.structure_net_qf2 = Int[params.l_s_t + params.l_out; 2^9; 2^7; 1] #Net structure Q function 2
params.init_range        = 0.001; #Range in which the weights are initialized
params.batch_size        = 16; #Size of number of timesteps for the Minibatch update
params.lr_pl             = 3e-4; #Learning rate for policy
params.lr_vf             = 3e-4; #Learning rate for value function
params.lr_qf             = 3e-4; #Learning rate for q function
params.σ                 = 0.05 * u_range; #Standard distribution for stochastic policy
params.n_traj            = 400; #Number of trajectories per iteration
params.n_iter            = 1500; #Number of training updates
params.buffer_size       = Int(floor(1e6 / params.batch_size)); #Maximum size of number of Minibatches in buffer
params.γ_disc            = 0.99; #Discount factor
params.λ_critic          = 0.0; #TD(λ) factor -0 is TD(0)-full history -1 is TD(1) only one step
params.λ_actor           = 0.95; #GAE factor
params.eps               = 0.2; #Clipping factor for PPO
params.δ                 = 0.01; #Delta Factor for TRPO update
params.cg_damping        = 0.1; #cg-damping factor for Fisher-Vector-Product
params.τ                 = 0.002; #Updating for running updates (mainly DDPG and SAC)
params.σ_off_policy      = 0.1; #SARSA regularization for off-policy q-training
params.clip_σ            = 0.5; #clipping for sarsa training
params.d_update          = 5; #Relation between q-function to policy training
#-------------------------------------------------------------------------------
