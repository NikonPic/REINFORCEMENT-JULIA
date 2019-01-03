#Do the Proximal Policy Optimization update:
function do_PPO_update(all_traj::Traj_Data, θa::Any, θv::Any, params::Params, opt_para_a, opt_para_v)
    for k = 1: all_traj.n_traj
        ĝ_a = grad_PPO(θa,params,all_traj.traj[k])

        ĝ_a2 = grad_PPO2(θa,params,all_traj.traj[k])

        figure(2)
        clf()
        plot(net_to_vec(ĝ_a))
        plot(net_to_vec(ĝ_a2))
        sleep(1)

        ĝ_v = grad_value(θv,params,all_traj.traj[k])
        Knet.update!(θa, ĝ_a, opt_para_a)
        Knet.update!(θv, ĝ_v, opt_para_v)
    end
    return θa
end


#Constructed to surrogate Objective to be optimized
@everywhere function loss_PPO(θa::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    eps    = params.eps
    σ      = params.σ

    #Init
    L_CLIP  = 0
    s_t = traj.s_t[1,:]

    #Hope: improved peroformance by going backwards...
    for i_t = 1:l_traj
        #Take from Struct
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t,:]
        r_t = traj.r_t[i_t]
        A_t = traj.A_t[i_t]
        p_a_t_old = traj.p_a_t[i_t,:]

        #Only for Segway
        s_mod = s_t
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_t[1:2]
        s_mod[3]     = sin(s_t[3])
        s_mod[4]     = cos(s_t[3])
        s_mod[5:end] = s_t[4:end]
        #-------------------------------=#


        out_t = mlp_nlayer_policy_norm(θa,s_mod,params) .+ 0.5 #Evaluate the Network
        p_a_t = get_π_prob(out_t,σ,a_t)

        ratio = prod(p_a_t ./ p_a_t_old) #Get the Ratio! prod(p_a_t./p_a_t_old)

        #Do the conditioned Clipping! -Hard-CLIP!
        clip_ratio = clamp(ratio, 1-eps, 1+eps)

        #Take the maximum value of both
        loc_L = max(ratio*A_t, clip_ratio*A_t) #Take the max of both!

        L_CLIP += loc_L
    end

    #Return the final Surrogate Objective
    return L_CLIP
end

@everywhere grad_PPO = grad(loss_PPO,1)


#Constructed to surrogate Objective to be optimized
@everywhere function loss_PPO2(θa::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    eps    = params.eps
    σ      = params.σ

    #Init
    L_CLIP  = 0
    s_t = traj.s_t[1,:]

    #Hope: improved peroformance by going backwards...
    for i_t = 1:l_traj
        #Take from Struct
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t,:]
        r_t = traj.r_t[i_t]
        A_t = traj.A_t[i_t]
        p_a_t_old = traj.p_a_t[i_t,:]

        #Only for Segway
        s_mod = s_t
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_t[1:2]
        s_mod[3]     = sin(s_t[3])
        s_mod[4]     = cos(s_t[3])
        s_mod[5:end] = s_t[4:end]
        #-------------------------------=#


        out_t = mlp_nlayer_policy_norm(θa,s_mod,params) #Evaluate the Network
        p_a_t = get_π_prob(out_t,σ,a_t)

        ratio = prod(p_a_t ./ p_a_t_old) #Get the Ratio! prod(p_a_t./p_a_t_old)

        #Do the conditioned Clipping! -Soft-CLIP!
        clip_ratio = 1 + eps*tanh.((ratio-1) / eps)

        #Take the maximum value of both
        loc_L = max(ratio*A_t, clip_ratio*A_t) #Take the max of both!

        L_CLIP += loc_L
    end

    #Return the final Surrogate Objective
    return L_CLIP
end

@everywhere grad_PPO2 = grad(loss_PPO2,1)


#Parallel PPO update
#-------------------------------------------------------------------------------
"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_PPO(params::Params,θa::Any,θv::Any,all_traj::Traj_Data,opt_para_a, opt_para_v)
        #Take all partial traj and shuffle their order!
        n_traj = all_traj.n_traj
        shuffle_arr = shuffle(1:n_traj)

        active_workers = workers()
        n_workers = length(active_workers)
        ĝ_a = copy(θa)
        #ĝ_v = copy(θv)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gradients, p, params, input_channel, output_channel)
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
                    @async put!(input_channel, (traj_local, θa, θv))
                end
            end

            #2. Take the results
            ĝ_a *= 0
            #ĝ_v *= 0
            for i_w = 1: n_workers
                i_true = ((i_parallel-1) * n_workers + i_w)
                if i_true <= n_traj
                    #Take from Channel
                    gradients = take!(output_channel)
                    #Reorder
                    ĝ_a += gradients[1]
                    #ĝ_v += gradients[2]
                end
            end

            #Update the Networks
            Knet.update!(θa, ĝ_a, opt_para_a)
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
                @async put!(input_channel, (ending_traj, θa, θv))
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)

        #Return the updated Networks
        return (θa, θv)
end


# Function running on the worker
@everywhere function distributed_gradients(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #Take the new Input
            input = take!(input_channel)

            #Reorder data:
            traj_local = input[1]
            θa         = input[2]
            θv         = input[3]

            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                      #gc()
                      break
            end

            #DO THE WORK:
            #-----------------------------------------------------------------------
            #Calculate the Policy Gradient using PPO
            ĝ_a_loc = grad_PPO(θa,params,traj_local)

            #Calculate the Value Function gradient using TD(1)
            ĝ_v_loc = 0#grad_value(θv,params,traj_local)
            #-----------------------------------------------------------------------

            put!(output_channel, (ĝ_a_loc, ĝ_v_loc))

        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end
    end

    return nothing
end

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------



#Constructed to surrogate Objective to be optimized
@everywhere function loss_PPO_combined(θ::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    eps    = params.eps
    σ      = params.σ

    #Init
    L_CLIP  = 0
    s_t = traj.s_t[1,:]

    #Hope: improved peroformance by going backwards...
    for i_t = 1:l_traj

        #1. Take from Struct
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t,:]
        r_t = traj.r_t[i_t]
        A_t = traj.A_t[i_t]
        R_t = traj.R_t[i_t]
        p_a_t_old = traj.p_a_t[i_t,:]
        val_t_old = traj.val_t[i_t,:]

        #2. Evaluate the Network
        out_t, val_t = mlp_nlayer(θ,s_t,params)

        #3. Get Policy update
        p_a_t = get_π_prob(out_t,σ,a_t) #Calculate Action Probability

        ratio = prod(p_a_t) / prod(p_a_t_old) #Get the Ratio!

        #Do the conditioned Clipping!
        clip_ratio = 1 + eps*tanh.((ratio-1) / eps)

        #Local Actor Error
        loc_AE = max(ratio*A_t, clip_ratio*A_t) #Take the max of both!

        #4. Get Value Update
        loc_VE = (val_t - R_t)^2


        L_CLIP += loc_AE + loc_VE
    end

    #Return the final Surrogate Objective
    return L_CLIP
end

@everywhere grad_PPO_combined = grad(loss_PPO_combined,1)


#Parallel PPO update
#-------------------------------------------------------------------------------
"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_PPO_combined(params::Params,θ::Any,all_traj::Traj_Data,opt_para)
        n_traj = all_traj.n_traj
        active_workers = workers()
        n_workers = length(active_workers)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Any}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gradients_combined, p, params, input_channel, output_channel)
            end
        end

        # Start optimizations
        n_parallel = Int(ceil(n_traj / n_workers))
        prog = Progress(n_traj,1)

        for i_parallel = 1 : n_parallel

            #1. Send the work tasks:
            for i_w = 1 : n_workers
                i_traj = ((i_parallel-1)*n_workers + i_w)
                if i_traj <= n_traj
                    traj_local = all_traj.traj[i_traj]
                    @async put!(input_channel, (traj_local, θ))
                end
            end

            #2. Take the results
            for i_w = 1: n_workers
                i_traj = ((i_parallel-1)*n_workers + i_w)
                if i_traj <= n_traj
                    #Take from Channel
                    ĝ = take!(output_channel)
                    #Update the Networks
                    Knet.update!(θ, ĝ, opt_para)
                end
            end
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
                @async put!(input_channel, (ending_traj, θ))
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)

        #Return the updated Networks
        return θ
end


# Function running on the worker
@everywhere function distributed_gradients_combined(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #Take the new Input
            input = take!(input_channel)

            #Reorder data:
            traj_local = input[1]
            θ          = input[2]

            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                      break
            end


            #DO THE WORK:
            #-----------------------------------------------------------------------
            #Calculate the Policy Gradient using PPO
            ĝ = grad_PPO_combined(θ,params,traj_local)
            #-----------------------------------------------------------------------

            put!(output_channel, ĝ)

        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end
    end

    return nothing
end




#Parallel PPO update
#-------------------------------------------------------------------------------
"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_PPO2(params::Params,θa::Any,θv::Any,all_traj::Traj_Data,opt_para_a, opt_para_v)
        #Take all partial traj and shuffle their order!
        n_traj = all_traj.n_traj
        shuffle_arr = shuffle(1:n_traj)

        active_workers = workers()
        n_workers = length(active_workers)
        ĝ_a = copy(θa)
        #ĝ_v = copy(θv)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gradients2, p, params, input_channel, output_channel)
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
                    @async put!(input_channel, (traj_local, θa, θv))
                end
            end

            #2. Take the results
            ĝ_a *= 0
            #ĝ_v *= 0
            for i_w = 1: n_workers
                i_true = ((i_parallel-1) * n_workers + i_w)
                if i_true <= n_traj
                    #Take from Channel
                    gradients = take!(output_channel)
                    #Reorder
                    ĝ_a += gradients[1]
                    #ĝ_v += gradients[2]
                end
            end

            #Update the Networks
            Knet.update!(θa, ĝ_a, opt_para_a)
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
                @async put!(input_channel, (ending_traj, θa, θv))
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)

        #Return the updated Networks
        return (θa, θv)
end


# Function running on the worker
@everywhere function distributed_gradients2(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #Take the new Input
            input = take!(input_channel)

            #Reorder data:
            traj_local = input[1]
            θa         = input[2]
            θv         = input[3]

            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                      #gc()
                      break
            end

            #DO THE WORK:
            #-----------------------------------------------------------------------
            #Calculate the Policy Gradient using PPO
            ĝ_a_loc = grad_PPO2(θa,params,traj_local)

            #Calculate the Value Function gradient using TD(1)
            ĝ_v_loc = 0#grad_value(θv,params,traj_local)
            #-----------------------------------------------------------------------

            put!(output_channel, (ĝ_a_loc, ĝ_v_loc))

        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end
    end

    return nothing
end
