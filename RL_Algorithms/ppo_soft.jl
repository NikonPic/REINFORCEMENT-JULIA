#This script executes the soft-PPO algorithm from Wilhelm


include("src/init.jl")

θa = init_θ_single_norm(params.structure_net_pl)
θv = init_θ_single_norm(params.structure_net_vf)
best_net = init_θ_single_norm(params.structure_net_pl)
res = zeros(params.n_iter)
prec = zeros(params.n_iter)
all_traj  = init_all_traj(params)


"""
Soft Proximal Policy Optimisation
"""
function ppo_soft(params::Params, θa::Any, θv::Any, res::Array{Float64,1}, prec::Array{Float64,1}, all_traj::Traj_Data, best_net::Any, i_start)
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

ppo_soft(params, θa, θv, res, prec, all_traj, best_net)
