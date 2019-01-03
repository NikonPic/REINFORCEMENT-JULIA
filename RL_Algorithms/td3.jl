#This file executes the TD3 algorithm: https://arxiv.org/pdf/1802.09477.pdf
#This is more advanced than the simple DDPG algorithm!!

include("src/init.jl")# Include all relevant functions

include("src/td3/episodes.jl")
include("src/td3/gradients.jl")
include("src/td3/updates.jl")


res  = zeros(params.n_iter)
prec = zeros(params.n_iter)

#=NIPS-challenge
include("src/NIPS2018.jl")
#Global Variable of Environment!
@everywhere @pyimport osim.env as osim
@everywhere env = osim.ProstheticsEnv(visualize=false)
env = osim.ProstheticsEnv(visualize=true)
@everywhere env[:change_model](model="2D", prosthetic=false, difficulty=0)
sleep(1)
# =#

#1. Randomly Initialize
#-------------------------------------------------------------
#1.1 First Critic
Qf_1         = init_θ_single_norm(params.structure_net_qf)
Qf_tar_1     = copy(Qf_1)
opt_para_qf1 = optimizers(Qf_1, Adam; lr = 0.5*params.lr_qf)

#1.1 Second Critic
Qf_2         = init_θ_single_norm(params.structure_net_qf2)
Qf_tar_2     = copy(Qf_2)
opt_para_qf2 = optimizers(Qf_2, Adam; lr = 2.0*params.lr_qf)


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


#using FileIO
#init_nets = load("Results/Videos/10_20_init_nets/mountaincar.jld2")
#Pl = init_nets["θa"]

#main function running DDPG
function TD3()
    weights_old = zeros(length(net_to_vec(Pl)),2)
    a_t_old = all_traj.traj[1].a_t

    #1.4 Init running mean std
    rms_all = init_rms(params.s_init)

    #Iterate
    for i_iter = 1:params.n_iter
        #Init Progress bar
        progress = Progress(params.n_iter,1)

        if floor(i_iter/100) == ceil(i_iter/100)
            #make_video(all_traj,params, i_iter)
            save_data(res, prec, Pl, Qf_1, all_traj, i_iter)
        end

        #2.Collect Data using current stochastic Policy
        #-------------------------------------------------------------
        res[i_iter] = parallel_episodes_td3(params, Pl, Qf_1, Qf_2, all_traj,rms_all)
        #res[i_iter], dump = parallel_episodes_skeleton(params,Pl,Pl_tar,all_traj,rms_all)

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
        if floor(i_iter/5) == ceil(i_iter/5)
            print("  new visualization: ")
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,Pl, Qf_1, res, prec/params.n_traj, Pl, Pl)
            #run_episode_skeleton_demo!(params,Pl,Pl_tar,all_traj.traj[1],rms_all,env)
        end
        print("  Result:  ", res[i_iter])
    end
end

TD3()
