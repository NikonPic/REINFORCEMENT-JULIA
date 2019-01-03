#This file will contain some demo functions for DDPG algorithms: https://arxiv.org/abs/1509.02971


#Function optimising the Q-function depending on the replay buffer
function q_training_all(buffer::Traj_Data,Q,Q_mod,θa,θa_mod, opt_para_a, opt_para_q, params::Params)
    train_size  = buffer.n_traj
    shuffle_arr = shuffle(1:train_size)
    #Sample from Batch:
    for i_true = 1:train_size
        i_traj = shuffle_arr[i_true]
        traj   = buffer.traj[i_traj]

        #get the local gradients
        ĝ_q = grad_Q(Q,Q_mod,θa,θa_mod,traj,params)
        ĝ_a = grad_DP(θa,θa_mod,Q,Q_mod,traj,params)

        #Update the evaluated networks:
        Knet.update!(θa, ĝ_a, opt_para_a)
        Knet.update!(Q,  ĝ_q, opt_para_q)

        #Update the target networks as well: (but slower)
        Q_mod  = params.τ .* Q  + (1 - params.τ) .* Q_mod
        θa_mod = params.τ .* θa + (1 - params.τ) .* θa_mod
    end
end

#Function calculationg the q-loss for updating
@everywhere function loss_Q(Q,Q_mod,θa,θa_mod,traj::Trajectory,params::Params)
    #
    len = traj.len
    lossQ = 0
    γ = params.γ_disc

    #Over the whole minibatch:
    for i_t = 1:len
        #Take from struct and precalculate
        r_t       = traj.r_t[i_t]
        s_t       = traj.s_t[i_t,:]
        s_t2      = traj.s_t2[i_t,:]
        a_t       = traj.a_t[i_t,:]

        out_t     = mlp_nlayer_policy_norm(θa_mod,s_t2,params)
        in_q_mod  = cat(s_t2,out_t; dims = 1)
        in_q      = cat(s_t,a_t; dims = 1)
        #Get both values
        y_i       = r_t + γ * mlp_nlayer_value_norm(Q_mod,in_q_mod,params)
        Q_out     = mlp_nlayer_value_norm(Q,in_q,params)
        #accumulate loss
        lossQ   += (y_i - Q_out)^2
    end

    #Return the final quadratic loss per timestep
    return (lossQ / len)
end

@everywhere grad_Q = grad(loss_Q,1)




#Function calculationg the q-loss for updating
@everywhere function loss_DP2(θa,θa_mod,Q,Q_mod,traj::Trajectory,params::Params)
    #general setup
    len = traj.len
    loss = 0

    #Over the whole minibatch:
    for i_t = 1:len
        #Take from struct and precalculate
        r_t    = traj.r_t[i_t]
        s_t    = traj.s_t[i_t,:]
        a_t    = traj.a_t[i_t,:]

        #get inputs
        out_p  = mlp_nlayer_policy_norm(θa,s_t,params)
        #in_q   = cat(s_t,out_p; dims = 1)
        #out_q  = mlp_nlayer_value_norm(Q,in_q,params)

        x  = cat(s_t,out_p; dims = 1) #stuff the input together

        #Here comes the evaluation as autograd has problems with calling the combined version
        #----------------------------------------
        for i = 1:3:length(Q)-2
            #1. Get the new summed input
            x = Q[i] * x

            #2. Calculate the mean and std of the layer
            x_mean, x_std = get_norm(x)

            #3. Normalize activation by applying the norm and mean
            x_norm = (x .- x_mean) / x_std

            #4. Reparametrize the output
            x = x_norm .* Q[i+1] + Q[i+2]

            #5. Apply the non-linearity
            x = relu.(x)
        end
        out_q = (Q[end-1] * x .+ Q[end])[1]
        #-----------------------------------------

        loss  += out_q
    end

    #Return the final loss
    return (loss / len)
end

@everywhere grad_DP2 = grad(loss_DP2,1)





#Function calculationg the q-loss for updating
@everywhere function grad_DP(θa,θa_mod,Q,Q_mod,traj::Trajectory,params::Params)
    #general setup
    len = traj.len
    γ = params.γ_disc
    ĝ_DP = net_to_vec(θa)*0

    #Over the whole minibatch:
    for i_t = 1:len
        #Take from struct and precalculate
        r_t    = traj.r_t[i_t]
        s_t    = traj.s_t[i_t,:]
        a_t    = traj.a_t[i_t,:]

        #get inputs
        out_t  = mlp_nlayer_policy_norm(θa,s_t,params)
        in_q   = cat(s_t, out_t; dims = 1)

        #Get the derivation of the Q-function depending on action:
        ∇Q_a = simple_grad_q_norm(out_t,s_t,Q,params)

        #Get the gradient of the policy depending on state
        ∇θ_s = zeros(length(net_to_vec(θa)),params.l_out)
        for d = 1:params.l_out
            ∇θ_s[:,d] = vec(net_to_vec(simple_grad_policy(θa,s_t,params,d)))
        end

        #accumulate loss
        ĝ_DP += ∇θ_s * ∇Q_a
    end

    #Return the final gradient
    return vec_to_net(ĝ_DP ./ len, θa)
end


#Get the gradient of the Policy towards input
@everywhere function eval_p(θa,s_t,params::Params,dim)
    out_t = mlp_nlayer_policy_norm(θa,s_t,params)[dim]
    return sum(out_t) #after some thinking...
end
@everywhere simple_grad_policy = grad(eval_p,1)


#Get the gradient of the q function after a
@everywhere function eval_Q(a_t,s_t,Q,params::Params)
    x  = cat(s_t,a_t; dims = 1) #stuff the input together

    #Here comes the evaluation as autograd has problems with calling the combined version
    #----------------------------------------
    for i = 1:2:length(Q)-2
        x = relu.(Q[i] * x .+ Q[i+1])
    end
    value    = (Q[end-1] * x .+ Q[end])[1]
    #----------------------------------------

    return value
end
@everywhere simple_grad_q = grad(eval_Q,1)



#Get the gradient of the q function after a
@everywhere function eval_Q_norm(a_t,s_t,Q,params::Params)
    x  = cat(s_t,a_t; dims = 1) #stuff the input together

    #Here comes the evaluation as autograd has problems with calling the combined version
    #----------------------------------------
    for i = 1:3:length(Q)-2
        #1. Get the new summed input
        x = Q[i] * x

        #2. Calculate the mean and std of the layer
        x_mean, x_std = get_norm(x)

        #3. Normalize activation by applying the norm and mean
        x_norm = (x .- x_mean) / x_std

        #4. Reparametrize the output
        x = x_norm .* Q[i+1] + Q[i+2]

        #5. Apply the non-linearity
        x = relu.(x)
    end

    value    = (Q[end-1] * x .+ Q[end])[1]
    #----------------------------------------
    return value
end
@everywhere simple_grad_q_norm = grad(eval_Q_norm,1)

"""
Function enabling parallel execution of the DPG gradients
"""
function parallel_DDPG_update(buffer::Traj_Data,Q,Q_mod,θa,θa_mod, opt_para_a, opt_para_q, params::Params)
        #Take all partial traj and shuffle their order!
        n_traj = buffer.n_traj
        shuffle_arr = shuffle(1:n_traj)

        active_workers = workers()
        n_workers = length(active_workers)
        ĝ_a = copy(θa)
        ĝ_q = copy(Q)

        # Create Channels storing results and starting values
        input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
        output_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gradients_dpg, p, params, input_channel, output_channel)
            end
        end

        # Start optimizations
        n_parallel = Int(ceil(n_traj / (2*n_workers)))
        prog = Progress(n_traj,1)

        for i_parallel = 1 : n_parallel

            #1. Send the work tasks:
            for i_w = 1 : n_workers
                i_true = ((i_parallel-1)*n_workers + i_w) #get true order
                if i_true <= n_traj
                    i_traj = shuffle_arr[i_true] #shuffle order
                    traj_local = buffer.traj[i_traj]
                    @async put!(input_channel, (traj_local, Q, Q_mod, θa, θa_mod) )
                end
            end

            #2. Take the results
            ĝ_q *= 0
            ĝ_a *= 0
            for i_w = 1: n_workers
                i_true = ((i_parallel-1)*n_workers + i_w)
                if i_true <= n_traj
                    #Take from Channel
                    gradients = take!(output_channel)
                    #Reorder
                    ĝ_q += gradients[1]
                    ĝ_a += gradients[2]
                end
            end
            ĝ_q /= n_workers
            ĝ_a /= n_workers

            #Update the evaluated networks:
            Knet.update!(θa, ĝ_a, opt_para_a)
            Knet.update!(Q,  ĝ_q, opt_para_q)

            #Update the target networks as well: (but slower)
            Q_mod  = params.τ .* Q  + (1 - params.τ) .* Q_mod
            θa_mod = params.τ .* θa + (1 - params.τ) .* θa_mod
        end

        # Tell workers that its done
        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        for j = 1 : n_workers
            @async put!(input_channel, (ending_traj, Q, Q_mod, θa, θa_mod))
        end

        #Close Channels
        sleep(0.1)
        close(input_channel)
        close(output_channel)

        #Return the updated Networks
        return (Q, Q_mod, θa, θa_mod)
end


# Function running on the worker
@everywhere function distributed_gradients_dpg(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)
    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #Take the new Input
            input = take!(input_channel)

            #Reorder data:
            traj_local = input[1]
            Q          = input[2]
            Q_mod      = input[3]
            θa         = input[4]
            θa_mod     = input[5]

            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                GC.gc()
                break
            end

            #DO THE WORK:
            #-------------------------------------------------------------------
            ĝ_q  = grad_Q(Q,Q_mod,θa,θa_mod,traj_local,params)
            #ĝ_a  = grad_DP(θa,θa_mod,Q,Q_mod,traj_local,params)
            ĝ_a = grad_DP2(θa,θa_mod,Q,Q_mod,traj_local,params)
            #-------------------------------------------------------------------

            put!(output_channel, (ĝ_q , ĝ_a))

        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end
    end

    return nothing
end
