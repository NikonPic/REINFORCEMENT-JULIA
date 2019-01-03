include("src/init.jl")# Include all relevant functions

include("src/sac/policy.jl")
include("src/sac/gradients.jl")
include("src/sac/updates.jl")


#If the architectur should be modified a bit, use this algorithm:
include("src/td3/episodes.jl")

#Pl  = init_θ_pol_SAC(params.structure_net_pl)
Pl  = init_θ_single(params.structure_net_pl)

Vf  = init_θ_single(params.structure_net_vf)
Qf1 = init_θ_single(params.structure_net_qf)
Qf2 = init_θ_single(params.structure_net_qf)
Vf_tar = copy(Vf)

best_net = copy(Pl)
res = zeros(params.n_iter)
prec = zeros(params.n_iter)
all_traj  = init_all_traj(params)




function soft_actor_critic(params::Params, Pl::Any, Vf::Any, Qf1::Any, Qf2::Any, Vf_tar::Any, res::Array{Float64,1},
    prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any)

    #Init Part:
    best_net = copy(Pl)
    min_cost = Inf
    #------------------------
    #Initialize the Policy
    opt_para_pl  = optimizers(Pl,  Adam; lr = params.lr_pl)
    opt_para_vf  = optimizers(Vf,  Adam; lr = params.lr_vf)
    opt_para_qf1 = optimizers(Qf1, Adam; lr = 2.0*params.lr_qf)
    opt_para_qf2 = optimizers(Qf2, Adam; lr = 0.5*params.lr_qf)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Init rms
    rms_all = init_rms(params.s_init)
    rms_loc = init_rms(params.s_init)

    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(Pl)),2)

    #buffer = Traj_Data(undef,0,0) #empty buffer
    buffer = Traj_Data()
    buffer.traj = Array{Trajectory,1}(undef,0)
    buffer.timesteps = 0
    buffer.n_traj = 0


    #Initialize with supervised Training of Value function:
    #parallel_episodes_SAC(params,Pl,Vf,all_traj,rms_all)
    parallel_episodes(params,Pl,Vf,all_traj,rms_all)


    #two_pass_variance!(all_traj, rms_all, params)

    mini_batch = build_minibatches(all_traj,params)

    pump_buffer!(buffer, mini_batch, params)

    for i_iter = 1:params.n_iter

        #Save data in intervalls
        if floor(i_iter/100) == ceil(i_iter/100)
            #make_video(all_traj,params, i_iter)
            save_data(res, prec, Pl, Vf, all_traj, i_iter)
        end

        #-----------------------------------------------------------------------
        #Solve it in Parallel!
        res[i_iter], prec[i_iter] = parallel_episodes_SAC(params,Pl,Vf,all_traj,rms_all)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params)

        #Pump the data into the buffer
        pump_buffer!(buffer, mini_batch, params)

        #Update all networks going over the collected data:
        println(" ")
        do_sac_update_parallel!(Pl, Qf1, Qf2, Vf, Vf_tar, buffer,
        params, opt_para_pl, opt_para_qf1, opt_para_qf2, opt_para_vf)

        if min_cost > res[i_iter]
            min_cost = res[i_iter]
            best_net = copy(Pl)
            println("new_best: ",min_cost)
        end

        #5.Visualize Progress
        ProgressMeter.update!(progress,i_iter) #update progress bar

        #6. Plot data of interest
        if floor(i_iter/5) == ceil(i_iter/5)
            print("  new visualization: ")
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,Pl, Qf1, res, prec/params.n_traj, Pl, Pl)
        end
        print("  Result:  ", res[i_iter])
    end
end

soft_actor_critic(params, Pl, Vf, Qf1, Qf2, Vf_tar, res, prec, all_traj, best_net)
