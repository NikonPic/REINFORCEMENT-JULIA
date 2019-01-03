#This file will contain some demo functions for DDPG algorithms:
include("src/init.jl")# Include all relevant functions
include("src/DDPG.jl") #Contains Deep deterministic Policy Gradient functions

"""
This file contains the script to execute the DDPG algorithm:
https://arxiv.org/abs/1509.02971
"""

res = zeros(params.n_iter)
prec = zeros(params.n_iter)


#1. Randomly Initialize
#-------------------------------------------------------------
#1.1 Actor
θa  = init_θ_single_norm(params.structure_net_pl)
θa_mod = copy(θa)
opt_para_a = optimizers(θa, Adam; lr = params.lr_pl)
ĝ_a = copy(θa)

#1.2 Critic
Q  = init_θ_single_norm(params.structure_net_qf)
Q_mod = copy(Q)
opt_para_q = optimizers(Q, Adam; lr = params.lr_qf)
ĝ_v = copy(Q)

#1.3 Replay Buffer and Latest Trajectories
buffer = Traj_Data()
buffer.traj = Array{Trajectory,1}(undef,0)
buffer.timesteps = 0
buffer.n_traj = 0

all_traj  = init_all_traj(params)


#main function running DDPG
function DDPG()
    weights_old = zeros(length(net_to_vec(θa)),2)
    a_t_old = all_traj.traj[1].a_t

    #1.4 Init running mean std
    rms_all = init_rms(params.s_init)

    #Iterate
    for i_iter = 1:params.n_iter
        #Init Progress bar
        progress = Progress(params.n_iter,1)

        if floor(i_iter/100) == ceil(i_iter/100)
            #make_video(all_traj,params, i_iter)
            save_data(res, prec, θa, Q, all_traj, i_iter)
        end

        #2.Collect Data using current stochastic Policy
        #-------------------------------------------------------------
        res[i_iter], prec[i_iter] = parallel_episodes(params,θa,θa,all_traj,rms_all)
        #(ĝ_a, ĝ_v, res[i_iter], prec[i_iter]) = parallel_episodes_skeleton(params,θa,θa,all_traj,rms_all)
        prec[i_iter] = loss_Q(Q,Q_mod,θa,θa_mod,all_traj.traj[1],params)

        #3.Pump data in Replay Buffer:
        #-------------------------------------------------------------
        #3.1 Generate minibatches from current data_batch
        mini_batch = build_minibatches(all_traj,params)
        #3.2 Pump the data into the buffer
        pump_buffer!(buffer,mini_batch,params)

        #4.Train all the stuff now:
        #-------------------------------------------------------------
        println(" ")
        @time parallel_DDPG_update(buffer,Q,Q_mod,θa,θa_mod,opt_para_a,opt_para_q,params)

        #5.Visualize Progress
        ProgressMeter.update!(progress,i_iter) #update progress bar
        if floor(i_iter/5) == ceil(i_iter/5)
            println("new visualization: ")
            a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,
            weights_old, i_iter,θa, Q, res, prec/params.n_traj, ĝ_a, ĝ_v)
        end
        println(res[i_iter])
    end
end

DDPG()
