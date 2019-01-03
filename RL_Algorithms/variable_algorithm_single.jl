include("src/init.jl")

"""
This file contains the script to execute learning containing
the Value and Critic function in one Network!
"""

θ = init_θ_combined(params.structure_net_pl)
res = zeros(params.n_iter)
precision = zeros(params.n_iter)
all_traj  = init_all_traj(params)

function actor_critic_combined(params::Params, θ::Any, res::Array{Float64,1}, precision::Array{Float64,1}, all_traj::Traj_Data)
    #Init Part:
    #------------------------
    #Initialize the Network
    ĝ = copy(θ)
    opt_para = optimizers(θ,   Adam; lr = params.lr_pl)

    #Init Progress bar
    progress = Progress(params.n_iter,1)

    # Preallocate trajectory
    rms_state = init_rms(params.s_init)

    a_t_old = all_traj.traj[1].a_t
    weights_old = zeros(length(net_to_vec(θ)),2)
    #FIM = Symmetric(eye(length(net_to_vec(θ))))*0 #preallocate FisherMatrix

    for i_iter = 1:params.n_iter
        ĝ *= 0

        #Make a Video
        if floor(i_iter/400) == ceil(i_iter/400)
            figure(1)
            #t_horizon = (all_traj.traj[1].len)*(params.dt)
            t_plot = 0:params.dt:params.t_horizon;
            x1 = all_traj.traj[1].s_t
            u1 = all_traj.traj[1].a_t'
            filename = string("Results/Videos/Actor_Critic_",Dates.today(),"_iter_",i_iter,".mp4")
            visualise(:SegwayRobot, x1', t_plot, filename; options = Dict(:U => u1, :size => (1920, 1080)))
        end

        #-----------------------------------------------------------------------
        #Solve it in Parallel! (How to do asynchron updates..?)
        ĝ, res[i_iter], precision[i_iter] = parallel_episodes_combined(params,θ,all_traj,rms_state)

        #Reorganice Data in minibatches
        mini_batch = build_minibatches(all_traj,params)

        #Do the Policy / and Value updates
        θ = parallel_PPO_combined(params, θ, mini_batch, opt_para)

        #Visualization_stuff
        #-----------------------------------------------------------------------
        ProgressMeter.update!(progress,i_iter) #update progress bar
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θ, θ, res, precision/params.n_traj, ĝ, ĝ)
        #-----------------------------------------------------------------------
    end
end

actor_critic_combined(params, θ, res, precision, all_traj)
#save_data(res, precision, θa, θv, all_traj)
