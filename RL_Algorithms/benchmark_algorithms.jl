#This Script evaluates multiple Algorithms for the desired Environment for training comparison!
include("src/init.jl")


"""
REINFORCE
"""
function reinforce(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    for i_iter = 1:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #Save data in intervalls
        if floor(i_iter/50) == ceil(i_iter/50)
            #make_video(all_traj,params, i_iter)
            #save_data(res, prec, θa, θv, all_traj, i_iter)
        end

        #-----------------------------------------------------------------------
        #Solve it in Parallel!
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Apply simple baseline
        simple_baseline!(all_traj::Traj_Data, params::Params)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params);

        #Apply the gradient
        ĝ_a = parallel_gradient(params, θa, mini_batch)
        Knet.update!(θa, ĝ_a, opt_para_a)

        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        #clean the workspace from junk
        @everywhere GC.gc(true)
    end
    return res
end


"""
ADVANTAGE ACTOR CRITIC
"""
function advantage_actor_critic(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    kl_means = zeros(params.n_iter)

    #saving old data to see differences
    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    for i_iter = 1:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #execute the parallel_episodes
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params)

        #First refit the value function for new data batch:
        θv = refit_VF(params, θv, mini_batch, opt_para_v)

        #Now apply the gae for the new data based on updated value function
        gae_all!(all_traj,θv,params)

        #Reorganice Data in minibatches as advantages have changed...
        mini_batch = build_minibatches(all_traj,params)

        #Get Policy Update
        ĝ_a = parallel_gradient(params, θa, mini_batch)
        Knet.update!(θa, ĝ_a, opt_para_a)

        #Get KL-div of last update step
        kl_means[i_iter] = kl_all(θa,all_traj::Traj_Data,params::Params)

        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        @everywhere GC.gc(true)
    end
    return res, kl_means
end


"""
TNPG
"""
function tnpg(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    #saving old data to see differences
    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    for i_iter = 1:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #execute the parallel_episodes
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params)

        #First refit the value function for new data batch:
        θv = refit_VF(params, θv, mini_batch, opt_para_v)

        #Now apply the gae for the new data based on updated value function
        gae_all!(all_traj,θv,params)

        #Reorganice Data in minibatches as advantages have changed...
        mini_batch = build_minibatches(all_traj,params)

        #Do the Policy / and Value updates
        ĝ_a = parallel_gradient(params, θa, mini_batch)
        θa  = do_TNPG_update(ĝ_a, mini_batch, θa, params)

        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        @everywhere GC.gc(true)
    end
    return res
end


"""
TRPO
"""
function trpo(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    #saving old data to see differences
    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    for i_iter = 1:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #execute the parallel_episodes
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params)

        #First refit the value function for new data batch:
        θv = refit_VF(params, θv, mini_batch, opt_para_v)

        #Now apply the gae for the new data based on updated value function
        gae_all!(all_traj,θv,params)

        #Reorganice Data in minibatches as advantages have changed...
        mini_batch = build_minibatches(all_traj,params)

        #Apply the gradient
        ĝ_a = parallel_gradient(params, θa, mini_batch)
        θa  = do_TRPO_update(ĝ_a, mini_batch, θa, params)

        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        @everywhere GC.gc(true)
    end
    return res
end


function ppo(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any, i_start)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    kl_means = zeros(params.n_iter)

    for i_iter = i_start:1:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #Save data in intervalls
        if floor(i_iter/50) == ceil(i_iter/50)
            #save_data(res, prec, θa, θv, all_traj, i_iter)
        end

        #-----------------------------------------------------------------------
        #Solve it in Parallel!
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params);

        #First refit the value function for new data batch:
        θv = refit_VF(params, θv, mini_batch, opt_para_v);

        #Now apply the gae for the new data based on updated value function
        gae_all!(all_traj,θv,params);

        #Reorganice Data in minibatches as advantages have changed...
        mini_batch = build_minibatches(all_traj,params)

        #Do the Policy / and Value updates
        (θa, dump) = parallel_PPO(params, θa, θv, mini_batch, opt_para_a, opt_para_v)

        #Get KL-div of last update step
        kl_means[i_iter] = kl_all(θa,all_traj::Traj_Data,params::Params)


        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        #clean the workspace from junk
        @everywhere GC.gc(true)
    end
    return res, kl_means
end


function ppo2(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any, i_start)
    #Init Part:
    best_net = copy(θa)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    ĝ_a = copy(θa)
    opt_para_a = optimizers(θa,   Adam; lr = params.lr_pl)

    #Initialize the Value Function
    ĝ_v = copy(θv)
    opt_para_v = optimizers(θv,   Adam; lr = params.lr_vf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θa)),2)

    kl_means = zeros(params.n_iter)

    for i_iter = i_start:params.n_iter
        ĝ_a *= 0
        ĝ_v *= 0

        #Save data in intervalls
        if floor(i_iter/50) == ceil(i_iter/50)
            #save_data(res, prec, θa, θv, all_traj, i_iter)
        end

        #-----------------------------------------------------------------------
        #Solve it in Parallel!
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θv,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params);

        #First refit the value function for new data batch:
        θv = refit_VF(params, θv, mini_batch, opt_para_v);

        #Now apply the gae for the new data based on updated value function
        gae_all!(all_traj,θv,params);

        #Reorganice Data in minibatches as advantages have changed...
        mini_batch = build_minibatches(all_traj,params)

        #Do the Policy / and Value updates
        (θa, dump) = parallel_PPO2(params, θa, θv, mini_batch, opt_para_a, opt_para_v)

        #Get KL-div of last update step
        kl_means[i_iter] = kl_all(θa,all_traj::Traj_Data,params::Params)


        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(θa)
            print(" new_best: ", round(min_cost, digits = 3))
        end

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/50) == ceil(i_iter/50)
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        #-----------------------------------------------------------------------

        #clean the workspace from junk
        @everywhere GC.gc(true)
    end
    return res, kl_means
end


"""
Time Delayed Deep Deterministic Policy Gradient
"""

include("src/td3/episodes.jl")
include("src/td3/gradients.jl")
include("src/td3/updates.jl")

#1. Randomly Initialize
#-------------------------------------------------------------

#1.3 Policy network
Pl           = init_θ_single_norm(params.structure_net_pl)
Pl_tar       = copy(Pl)
opt_para_pl  = optimizers(Pl, Adam; lr = params.lr_pl)


#1.4 Replay Buffer and Latest Trajectories
buffer           = Traj_Data()
buffer.traj      = Array{Trajectory,1}(undef,0)
buffer.timesteps = 0
buffer.n_traj    = 0

#1.5 Buffer for the most recent trajectories
all_traj  = init_all_traj(params)


#main function running DDPG
function td3(Pl::Any, res)

    #1.1 First Critic
    Qf_1         = init_θ_single_norm(params.structure_net_qf)
    Qf_tar_1     = copy(Qf_1)
    opt_para_qf1 = optimizers(Qf_1, Adam; lr = 0.5*params.lr_qf)

    #1.1 Second Critic
    Qf_2         = init_θ_single_norm(params.structure_net_qf2)
    Qf_tar_2     = copy(Qf_2)
    opt_para_qf2 = optimizers(Qf_2, Adam; lr = 2.0*params.lr_qf)

    #1.3 Thrid Actor
    #Pl           = init_θ_single_norm(params.structure_net_pl)
    Pl_tar       = copy(Pl)
    opt_para_pl  = optimizers(Pl, Adam; lr = params.lr_pl)


    weights_old = zeros(length(net_to_vec(Pl)),2)
    a_t_old = all_traj.traj[1].a_t

    #1.4 Init running mean std
    rms_all = init_rms(params.s_init)

    #Iterate
    for i_iter = 1:params.n_iter
        #Init Progress bar
        progress = Progress(params.n_iter,1)

        if floor(i_iter/100) == ceil(i_iter/100)
            #save_data(res, prec, Pl, Qf_1, all_traj, i_iter)
        end

        #2.Collect Data using current stochastic Policy
        #-------------------------------------------------------------
        #res[i_iter] = parallel_episodes_td3(params, Pl, Qf_1, Qf_2, all_traj,rms_all)
        res[i_iter], dump = parallel_episodes(params,Pl,Pl_tar,all_traj,rms_all)

        #3.Pump data in Replay Buffer:
        #-------------------------------------------------------------
        #3.1 Generate minibatches from current data_batch
        mini_batch = build_minibatches(all_traj,params)
        #3.2 Pump the data into the buffer
        pump_buffer!(buffer,mini_batch,params)

        #4.Train all the stuff now:
        #-------------------------------------------------------------
        do_td3_update_parallel!(Qf_1, Qf_tar_1, Qf_2, Qf_tar_2, Pl, Pl_tar, buffer, params, opt_para_pl, opt_para_qf1, opt_para_qf2)

        #5.Visualize Progress
        ProgressMeter.update!(progress,i_iter) #update progress bar

        #6. Plot data of interest
        if floor(i_iter/50) == ceil(i_iter/50)
            print("  new visualization: ")
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,Pl, Qf_1, res, prec/params.n_traj, Pl, Pl)
        end
        print("  Result:  ", round(res[i_iter], digits = 3))
    end
    return res
end


#"""
#-==============================================================================
#                                 Get running
#-==============================================================================
#"""
using FileIO

testruns      = 1
#params.n_iter = 20

res      = zeros(params.n_iter)
prec     = zeros(params.n_iter)
all_traj = init_all_traj(params)
best_net = init_θ_single_norm(params.structure_net_pl)



filename = "ballbot"
openfile = string("Results/10_20_init_nets/",filename,".jld2")
init_nets = load(openfile)
result = zeros(6,params.n_iter)
zw_save = "zw_save.jld2"




RESULT = []
result = zeros(6,params.n_iter)
function evaluate_all_methods()

    for i_all = 1: testruns
        result = zeros(6,params.n_iter)
        kl_res = zeros(6,params.n_iter)
        fig_label  = string("All Methods ", i_all)
        fig_label2 = string("All Methods kl ", i_all)
        figure(fig_label)
        clf()

        println("REINFORCE")
        init_nets1 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets1["θa"])
        θv  = copy(init_nets1["θv"])
        result[1,:] = reinforce(params, θa, θv, zeros(params.n_iter), prec, all_traj, best_net)
        figure(fig_label)
        plot(result[1,:], label="REINFORCE")
        newfile = string("Results/REINFORCE_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result


        println("A2C")
        init_nets2 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets2["θa"])
        θv  = copy(init_nets2["θv"])
        result[2,:], kl_res[2,:] = advantage_actor_critic(params, θa, θv, zeros(params.n_iter), prec, all_traj, best_net)
        figure(fig_label)
        plot(result[2,:], label="A2C")
        legend()
        newfile = string("Results/A2C_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result, kl_res

        println("TNPG")
        init_nets3 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets3["θa"])
        θv  = copy(init_nets3["θv"])
        result[3,:] = tnpg(params, θa, θv, zeros(params.n_iter), prec, all_traj, best_net)
        figure(fig_label)
        plot(result[3,:], label="TNPG")
        legend()
        newfile = string("Results/TNPG_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result

        println("TRPO")
        init_nets4 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets4["θa"])
        θv  = copy(init_nets4["θv"])
        result[4,:] = trpo(params, θa, θv, zeros(params.n_iter), prec, all_traj, best_net)
        figure(fig_label)
        plot(result[4,:], label="TRPO")
        legend()
        newfile = string("Results/TRPO_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result

        println("PPO-Hard")
        init_nets5 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets5[1])
        θv  = copy(init_nets5[2])
        result[5,:], kl_res[5,:] = ppo(params, θa, θv, res5 ,prec, all_traj, best_net, 1024)
        figure(fig_label)
        plot(result[5,:], label="PPO-Hard")
        legend()
        figure(fig_label2)
        plot(kl_res[5,:], label="PPO-Hard")
        legend()
        newfile = string("Results/PPO1_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result, kl_res

        println("PPO-Soft")
        init_nets6 = load(openfile)
        θa = init_θ_single_norm(params.structure_net_pl)
        θv = init_θ_single_norm(params.structure_net_vf)
        θa  = copy(init_nets5[1])
        θv  = copy(init_nets5[2])
        result[6,:], kl_res[6,:] = ppo2(params, θa, θv, res6, prec, all_traj, best_net, 1024)
        figure(fig_label)
        plot(result[6,:], label="PPO-Soft")
        legend()
        newfile = string("Results/PPO2_",i_all,"_data.jld2")
        @save newfile θa, θv, all_traj, best_net, result, kl_res

        savefile = string("Results/result",Dates.today(),"_iter_",i_all,"_zw.jld2")
        @save savefile result

        append!(RESULT,result)
    end
end

evaluate_all_methods()

savefile = string("Results/Actor_Critic_",Dates.today(),"done.jld2")
@save savefile RESULT
