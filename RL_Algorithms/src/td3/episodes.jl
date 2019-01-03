#Functions focusing on the episode execution: Value and Actions diverse
#-------------------------------------------------------------------------------
@everywhere function run_episode_td3!(params::Params,Pl::Any, Qf_1::Any, Qf_2::Any, traj::Trajectory, rms::RunMeanStd)
    dt          = params.dt
    t_horizon   = params.t_horizon
    s_init      = params.s_init + 0.01*sum(params.σ)*randn(length(params.s_init))
    l_traj      = params.l_traj
    σ           = params.σ
    γ           = params.γ_disc
    u_range     = params.u_range
    l_out       = length(u_range)
    #s_init[1:2] = 0.2 .* randn(2)
    #s_init[3]   = 1.2 .* randn()

    #Initialize :
    s_true = zeros(length(s_init)-1)
    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0
    rand_vec = zeros(l_out)


    for i_t = 1:l_traj
        #Normalize the Input
        #s_norm = normalize_rms(rms,s_t1) #select for normalization
        s_norm = s_t1 #select for unnormalized output

        #Evaluate the Network Output
        out_t = mlp_nlayer_policy_norm(Pl,s_norm,params)

        #Sample:
        rand_vec = randn(l_out)

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ, rand_vec)

        in_q   = cat(s_norm, loc_a_t; dims = 1)
        val_t1 = mlp_nlayer_value_norm(Qf_1 ,in_q, params)[1]
        val_t2 = mlp_nlayer_value_norm(Qf_2 ,in_q, params)[1]

        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call(s_t1, loc_a_t, params)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t
        traj.p_a_t[i_t,:]   = p_a_t
        traj.r_t[i_t]       = loc_r_t * 0.01
        traj.val_t[i_t]     = val_t1
        traj.A_t[i_t]       = val_t2
        traj.out_t[i_t,:]   = out_t
        traj.len            = i_t
        totalr             += loc_r_t

        #Override the state:
        s_t1 .= s_t2;

        #Check if the terminal condition is fullfilled and break
        if terminal == 1
                break
        end

    end

    return totalr / traj.len
end


#Get the required action
@everywhere function sample_action(μ, σ, rand_vec)
    μ = convert(Array{Float64}, μ)
    return  μ .+ rand_vec .* σ
end



#Functions for parallel execution:
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_episodes_td3(params::Params,Pl::Any,Qf_1::Any, Qf_2::Any, all_traj::Traj_Data,rms_state::RunMeanStd)
        n_traj = params.n_traj
        active_workers = workers()

        # Create Channels storing results and starting valus
        s_init_channel = RemoteChannel(()->Channel{Vector{Float64}}(length(workers())+1)) # will contain s_init
        result_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain traj and gradients

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_episodes_td3, p, params, Pl, Qf_1, Qf_2, rms_state, s_init_channel, result_channel)
            end
        end

        # Start optimizations
        for i_traj = 1:n_traj
                s_init_local = params.s_init
                @async put!(s_init_channel, s_init_local)
        end

        # Collect the resulting data
        res  = 0
        #prog = Progress(n_traj,1)
        for i_traj = 1:n_traj
            #Take from Channel:
            result = take!(result_channel)
            #Oder the Tuple:
            all_traj.traj[i_traj] = result[1]
            res                  += result[2]
        end

        # Tell workers that its done
        for j = 1:length(workers())
                NaN_mat = zeros(length(params.s_init))
                fill!(NaN_mat,NaN)
                @async put!(s_init_channel, NaN_mat)
        end

        #Close Channels
        sleep(0.1)
        close(s_init_channel)
        close(result_channel)

        #Return the Gradients:
        return res / n_traj
end



# Function running on the worker
@everywhere function distributed_episodes_td3(params::Params, Pl::Any, Qf_1::Any, Qf_2::Any, rms_state::RunMeanStd,
         s_init_channel::RemoteChannel, result_channel::RemoteChannel)

    # Preallocate trajectory
    traj = init_traj(params,params.l_traj)
    s_init_local = zeros(length(params.s_init))
    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            s_init_local .= take!(s_init_channel)
            if (sum(isnan, s_init_local) > 0)
                break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #DO THE WORK:
        #-----------------------------------------------------------------------
        res = run_episode_td3!(params, Pl, Qf_1, Qf_2, traj, rms_state)
        #Discount the rewards for supervised Training of Value function
        traj.R_t = vec(discount(traj.r_t, params.γ_disc))
        #-----------------------------------------------------------------------

        put!(result_channel, (traj, res))
    end

    return nothing
end
