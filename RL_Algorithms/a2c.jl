#This script executes the Advantage Adctor Critic algorithm from Minh


include("src/init.jl")

θa = init_θ_single_norm(params.structure_net_pl)
θv = init_θ_single_norm(params.structure_net_vf)
best_net = init_θ_single_norm(params.structure_net_pl)
res = zeros(params.n_iter)
prec = zeros(params.n_iter)
all_traj  = init_all_traj(params)


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

advantage_actor_critic(params, θa, θv, res, prec, all_traj, best_net)
