using Random
#Implementing the general Advantage Estimation according to https://arxiv.org/abs/1506.02438

#Return the Generalized Advantage Estimation:
@everywhere function gae!(params::Params,traj::Trajectory)
    l_traj = traj.len
    γ      = params.γ_disc
    λ      = params.λ_actor

    Val = traj.val_t
    r_t = traj.r_t

    #Calculate the local TD-error:
    delta_V_i = zeros(l_traj)
    for i_t = 1:l_traj-1
        delta_V_i[i_t] = -Val[i_t] + r_t[i_t] + γ*Val[i_t+1]
    end

    #Discount the TD-Error as the Advantage
    A_t = vec(discount(delta_V_i,λ*γ))

    #Write result in trajectory
    traj.A_t = A_t
end

#Parallel PPO update
#-------------------------------------------------------------------------------
"""
Function training the Value function on the last batch of data in parallel
"""
function refit_VF(params::Params,θv::Any,all_traj::Traj_Data, opt_para_v)
        #Take all partial traj and shuffle their order!
        n_traj = all_traj.n_traj
        shuffle_arr = Random.shuffle(1:n_traj)

        active_workers = workers()
        n_workers = length(active_workers)
        ĝ_v = copy(θv)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Array{Any,1}}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_vf_training, p, params, input_channel, output_channel)
            end
        end

        # Start optimizations
        n_parallel = Int(ceil(n_traj / n_workers))
        prog = Progress(n_traj,1)

        for i_parallel = 1 : n_parallel

            #1. Send the work tasks:
            for i_w = 1 : n_workers
                i_true = ((i_parallel-1)*n_workers + i_w) #get true order
                if i_true <= n_traj
                    i_traj = shuffle_arr[i_true] #shuffle order
                    traj_local = all_traj.traj[i_traj]
                    @async put!(input_channel, (traj_local, θv))
                end
            end

            #2. Take the result
            ĝ_v *= 0
            for i_w = 1: n_workers
                i_true = ((i_parallel-1)*n_workers + i_w)
                if i_true <= n_traj
                    #Take from Channel
                    ĝ_v += take!(output_channel)
                end
            end

            #Update the Network
            Knet.update!(θv, ĝ_v, opt_para_v)
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
                @async put!(input_channel, (ending_traj, θv))
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)

        #Return the updated Network
        return θv
end


# Function running on the worker
@everywhere function distributed_vf_training(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #Take the new Input
            input = take!(input_channel)

            #Reorder data:
            traj_local = input[1]
            θv         = input[2]

            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                      break
            end

            #DO THE WORK:
            #-----------------------------------------------------------------------
            #Calculate the Value Function gradient using TD(1)
            ĝ_v_loc = grad_value(θv,params,traj_local)
            #-----------------------------------------------------------------------

            put!(output_channel, ĝ_v_loc)

        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            println(traj_local)
            break
        end
    end

    return nothing
end





#Function applying general advantage estimation for each trajectory
function gae_all!(all_traj::Traj_Data,θv::Any,params::Params)
        #Take all partial traj and shuffle their order!
        n_traj = all_traj.n_traj

        active_workers = workers()
        n_workers = length(active_workers)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Trajectory}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Trajectory}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gae, p, params, input_channel, output_channel, θv)
            end
        end

        # Start optimizations
        n_parallel = Int(ceil(n_traj / n_workers))
        prog       = Progress(n_traj,1)

        # Start optimizations
        for i_traj = 1:n_traj
                traj_local = all_traj.traj[i_traj]
                @async put!(input_channel, traj_local)
        end

        for i_traj = 1:n_traj
                #Take from Channel:
                all_traj.traj[i_traj] = copy_traj(take!(output_channel),params)
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
                @async put!(input_channel, ending_traj)
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)
end


#Function running on the worker
@everywhere function distributed_gae(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel, θv)

   traj_local = init_traj(params,params.l_traj)
   # Start loop where work is done
   while true
       # break loop if NaN state received
       try
           #Take the new Input
           traj_local = copy_traj(take!(input_channel),params)

           #Check if u reached final
           if (sum(isnan, traj_local.r_t) > 0)
                     break
           end

           #DO THE WORK:
           #-----------------------------------------------------------------------
           #Recalculate the values for the states
           generate_values!(traj_local,params,θv)
           #recalculate the gae
           gae!(params,traj_local)
           #-----------------------------------------------------------------------

           put!(output_channel,copy_traj(traj_local,params))

       catch
           println(traj_local)
           println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
           break
       end
   end

   return nothing
end

#Function calculating the values for each state in a trajectory
@everywhere function generate_values!(traj::Trajectory,params::Params,θv::Any)
    for i_t = 1:traj.len
        s_t = traj.s_t[i_t,:]

        #Only for Segway
        s_mod = s_t
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_t[1:2]
        s_mod[3]     = sin(s_t[3])
        s_mod[4]     = cos(s_t[3])
        s_mod[5:end] = s_t[4:end]
        #-------------------------------=#

        val_t = mlp_nlayer_value_norm(θv,s_mod,params)[1]
        traj.val_t[i_t] = val_t
    end
end
