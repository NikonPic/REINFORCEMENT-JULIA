#Init the Net
function init_θ_pol_SAC(net_s::Array{Int,1})
    """
    This network will contain weights for actions and log_std
    """
    x      = net_s[1]
    hidden = net_s[2:end-1]
    ysize  = net_s[end]

    w = []
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, xavier(y, 1))
        x = y
    end
    push!(w, xavier(ysize, x)) #Weights outputting the mean
    push!(w, zeros(ysize, 1))

    push!(w, xavier(ysize, x)) #Weights outputting the log_std
    push!(w, zeros(ysize, 1))

    return w
end


#Init the Net
function init_θ_pol_SAC_norm(net_s::Array{Int,1})
    """
    This network will contain weights for actions and log_std
    """
    x      = net_s[1]
    hidden = net_s[2:end-1]
    ysize  = net_s[end]

    w = []
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, ones(y))
        push!(w, zeros(y))
        x = y
    end
    push!(w, xavier(ysize, x)) #Weights outputting the mean
    push!(w, zeros(ysize, 1))

    push!(w, xavier(ysize, x)) #Weights outputting the log_std
    push!(w, zeros(ysize, 1))

    return w
end



#Evaluate the Policy-Network
@everywhere function mlp_nlayer_policy_SAC(w::Any,x::AbstractVector)
    """
    Evaluation of the network
    """

    LOG_SIG_MAX = 2.0
    LOG_SIG_MIN = -20.0

    #Go over the hidden layers:
    for i = 1:2:length(w)-4
        x = relu.(w[i] * x .+ w[i+1])
    end

    mu_act   = vec(w[end-3] * x .+ w[end-2]) # without tanh here
    log_std  = vec(w[end-1] * x .+ w[end])

    std = exp.(log_std)

    return mu_act, std
end


#Evaluate the Policy-Network
@everywhere function mlp_nlayer_policy_SAC_norm(w::Any,x::AbstractVector)
    """
    Evaluation of the network
    """

    LOG_SIG_MAX = 2.0
    LOG_SIG_MIN = -20.0

    #Go over the hidden layers:
    for i = 1:3:length(w)-4
        #1. Get the new summed input
        x = w[i] * x

        #2. Calculate the mean and std of the layer
        x_mean, x_std = get_norm(x)

        #3. Normalize activation by applying the norm and mean
        x_norm = (x .- x_mean) / x_std

        #4. Reparametrize the output
        x = x_norm .* w[i+1] + w[i+2]

        #5. Apply the non-linearity
        x = relu.(x)
    end

    mu_act   = vec(w[end-3] * x .+ w[end-2]) # without tanh here
    log_std  = vec(w[end-1] * x .+ w[end])

    std = exp.(log_std)

    return mu_act, std
end


#Get the required action
@everywhere function sample_action_SAC(μ::AbstractVector, σ::AbstractVector, rand_vec::AbstractVector, params::Params)
    u_range = params.u_range
    μ       = convert(Array{Float64}, μ)
    sample  = μ .+ rand_vec .* σ
    actions = u_range.*tanh.(sample)
    #Return both values for probability calculation:
    return actions, sample
end


#Get the log(pi(a|s))
@everywhere function logpdf_SAC(a_t,out_t,std)
    fac = Float64(-log(sqrt(2pi)))
    r = (a_t-out_t) ./ std
    return -r.* r.* Float64(0.5) - (log.(std)) .+ fac
end



@everywhere function run_episode_SAC!(params::Params,Pl::Any, Vf::Any, traj::Trajectory, rms::RunMeanStd)
    dt          = params.dt
    t_horizon   = params.t_horizon
    s_init      = params.s_init + 0.1*sum(params.σ)*randn(length(params.s_init))
    l_traj      = params.l_traj
    σ           = params.σ
    u_range     = params.u_range
    l_out       = length(u_range)

    #Initialize :
    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0

    for i_t = 1:l_traj
        #Normalize the Input
        #s_norm = normalize_rms(rms,s_t1) #select for normalization
        s_norm = s_t1 #select for unnormalized output

        #Evaluate the Network Output
        out_t, std = mlp_nlayer_policy_SAC_norm(Pl,s_norm)
        val_t      = mlp_nlayer_value_norm(Vf,s_norm,params)

        #Get Action with stochastic policy:
        rand_vec         = randn(l_out)
        loc_a_t, sample  = sample_action_SAC(out_t, std, rand_vec, params)
        p_a_t            = get_π_prob(out_t,std,sample)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call(s_t1,loc_a_t,params)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t
        traj.p_a_t[i_t,:]   = p_a_t
        traj.r_t[i_t]       = 0.01 * loc_r_t  #* -1 #REWARD NEGATED HERE!!!!
        traj.out_t[i_t,:]   = out_t
        traj.val_t[i_t]     = val_t
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


"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_episodes_SAC(params::Params,Pl::Any,Vf::Any,all_traj::Traj_Data,rms_state::RunMeanStd)
        n_traj = params.n_traj
        active_workers = workers()

        # Create Channels storing results and starting valus
        s_init_channel = RemoteChannel(()->Channel{Vector{Float64}}(length(workers())+1)) # will contain s_init
        result_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain traj and gradients

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_episodes_SAC, p, params, Pl, Vf, rms_state, s_init_channel, result_channel)
            end
        end

        # Start optimizations
        for i_traj = 1:n_traj
                s_init_local = params.s_init
                @async put!(s_init_channel, s_init_local)
        end

        # Collect the resulting data
        prec = 0
        res = 0
        prog = Progress(n_traj,1)
        for i_traj = 1:n_traj
                #Take from Channel:
                result = take!(result_channel)
                #Oder the Tuple:
                all_traj.traj[i_traj] = result[1]
                res  += result[2]
                prec += result[3]
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

        #Return the Feedback
        return (res, prec)
end


# Function running on the worker
@everywhere function distributed_episodes_SAC(params::Params, Pl::Any, Vf::Any, rms_state::RunMeanStd, s_init_channel::RemoteChannel, result_channel::RemoteChannel)

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
        res = run_episode_SAC!(params,Pl, Vf, traj, rms_state)

        #Collect Precision
        prec = loss_value(Vf,params,traj)
        #-----------------------------------------------------------------------

        put!(result_channel, (traj, res, prec))
    end

    return nothing
end
