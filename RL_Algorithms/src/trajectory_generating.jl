# This file contains the functions to execute the normal episode:

#Functions focusing on the episode execution: Value and Actions diverse
#-------------------------------------------------------------------------------
@everywhere function run_episode!(params::Params,θa::Any, θv::Any, traj::Trajectory, rms::RunMeanStd)
    dt          = params.dt
    t_horizon   = params.t_horizon
    s_init      = params.s_init + 0.1*sum(params.σ)*randn(length(params.s_init))
    l_traj      = params.l_traj
    σ           = params.σ
    γ           = params.γ_disc
    u_range     = params.u_range
    l_out       = length(u_range)
    s_init[1:2] = 0.2 .* randn(2)
    s_init[3]   = 1.2 .* randn()

    #Initialize :
    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0
    rand_vec = zeros(l_out)


    for i_t = 1:l_traj

        #select for unnormalized output
        s_norm = s_t1


        #Only for Segway
        s_mod = s_norm
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_norm[1:2]
        s_mod[3]     = sin(s_norm[3])
        s_mod[4]     = cos(s_norm[3])
        s_mod[5:end] = s_norm[4:end]
        #-------------------------------=#


        #Evaluate the Network Output
        out_t = mlp_nlayer_policy_norm(θa,s_mod,params)
        val_t = mlp_nlayer_value_norm(θv,s_mod,params)[1]

        #Sample:
        rand_vec = randn(l_out)

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ, rand_vec)
        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call(s_t1,loc_a_t,params)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t
        traj.p_a_t[i_t,:]   = p_a_t
        traj.r_t[i_t]       = loc_r_t*0.01
        traj.val_t[i_t]     = val_t
        traj.out_t[i_t,:]   = out_t
        traj.len            = i_t
        totalr             += loc_r_t

        #Check if the terminal condition is fullfilled and break
        if terminal == 1
            traj.s_t[i_t,:] .= s_t1
            totalr          += (1/(1-γ)) * traj.r_t[traj.len]
            break
        end

        #Override the state:
        s_t1 .= s_t2;

    end

    #Return the average cost of the trajectory
    return totalr / (traj.len*2)
end



#Get the required action
@everywhere function sample_action(μ, σ, rand_vec)
    μ = convert(Array{Float64}, μ)
    return  μ .+ rand_vec .* σ
end



#Interact with environment
@everywhere function environment_call(s_t1::Array{Float64,1},a_t::Array{Float64,1}, params::Params)
    #Ineract with environment:
    s_t2 = rk4_func(s_t1, a_t, params.dt)

    #Claim the reward:
    r_t, terminal = cost_func(s_t2,a_t)

    return s_t2, r_t, terminal
end



#Functions focusing on the episode execution: Value and Actions combined
#-------------------------------------------------------------------------------
@everywhere function run_episode!(params::Params,θ::Any, traj::Trajectory, rms::RunMeanStd)
    dt         = params.dt;
    t_horizon  = params.t_horizon;
    s_init     = params.s_init + 0.1*sum(params.σ)*rand(length(params.s_init))
    l_traj     = params.l_traj;
    σ          = params.σ;
    u_range    = params.u_range;
    s_init[3]  = randn()

    #Initialize :
    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0

    for i_t = 1:l_traj
        #Normalize the Input
        s_norm = normalize_rms(rms,s_t1)
        #s_norm = s_t1

        #Evaluate the Network Output
        out_t, val_t = mlp_nlayer(θ,s_norm,params)

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ)
        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call(s_t1,loc_a_t,params)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t
        traj.p_a_t[i_t,:]   = p_a_t
        traj.r_t[i_t]       = loc_r_t*0.01
        traj.val_t[i_t]     = val_t
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
