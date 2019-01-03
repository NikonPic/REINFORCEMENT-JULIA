#containing functions trying to applicate the "Soft Actor Critic" from https://arxiv.org/abs/1801.01290
#First we need a policy with a variable gaussian distribution:

@everywhere function grad_all_networks(traj::Trajectory, Vf::Any, Vf_tar::Any, Qf1::Any, Qf2::Any, Pl::Any, params::Params)
    """
    This function determines the gradients for all networks over a Trajectory
    """
    #Take from struct
    len = traj.len
    a_dim = length(params.u_range)
    s_dim = length(params.s_init)
    γ = params.γ_disc

    #Preallocate target data:
    Vf_st_arr      = zeros(len)
    Vf_tar_st2_arr = zeros(len)
    pre_log_pi_arr = zeros(len)
    log_pi_arr     = zeros(len)
    Qf_st_new_arr  = zeros(len)
    Qf1_st_new_arr = zeros(len)
    rand_arr       = zeros(len,a_dim)
    in_q_arr       = zeros(len,s_dim+a_dim)

    """
    Precalculation to generate targets for the loss functions:
    This has to be done with the latest Versions of networks
    """
    for i_t = 1: traj.len
        #Take from struct
        s_t   = traj.s_t[i_t,:] #state
        s_t2  = traj.s_t2[i_t,:] #new state
        a_t   = traj.a_t[i_t,:] #old actions
        p_a_t = traj.p_a_t[i_t,:] #old probabilities
        in_q  = cat(s_t,a_t; dims=1)

        # Pre-evaluation
        Vf_st           = mlp_nlayer_value_norm(Vf,     s_t, params)[1]
        Vf_tar_st2      = mlp_nlayer_value_norm(Vf_tar, s_t2,params)[1]


        #-----------------------------------------------------------------------
        # Generate new acions: MAXIMUM ENTROPY
        #=
        Pl_st, std1     = mlp_nlayer_policy_SAC_norm(Pl,s_t)
        rand_vec        = randn(a_dim)
        a_t_new, sample = sample_action_SAC(Pl_st, std1, rand_vec, params)
        log_pi          = sum(logpdf_SAC(sample,Pl_st,std1))
        =#

        #Generate new actions: CLASSIC
        Pl_st           = mlp_nlayer_policy_norm(Pl,s_t,params)
        rand_vec        = randn(a_dim)
        a_t_new         = sample_action(Pl_st, params.σ, rand_vec)
        p_a_t_new       = get_π_prob(Pl_st,params.σ,a_t_new)
        log_pi          = sum(log.(p_a_t_new))
        #-----------------------------------------------------------------------


        in_q_new        = cat(s_t,a_t_new; dims = 1)
        Qf1_st_new      = mlp_nlayer_value_norm(Qf1,in_q_new,params)[1]
        Qf2_st_new      = mlp_nlayer_value_norm(Qf2,in_q_new,params)[1]

        #Save target data in Arrays:
        Vf_st_arr[i_t]      = Vf_st
        Vf_tar_st2_arr[i_t] = Vf_tar_st2
        pre_log_pi_arr[i_t] = sum(log.(p_a_t))
        log_pi_arr[i_t]     = log_pi
        Qf_st_new_arr[i_t]  = max(Qf1_st_new,Qf2_st_new) #using best of q-learning
        Qf1_st_new_arr[i_t] = Qf1_st_new #using best of q-learning
        rand_arr[i_t,:]     = rand_vec
        in_q_arr[i_t,:]     = in_q
    end

    """
    Gradient of Value function:
    """
    ĝ_vf = ∇_Vf(Vf, traj, Qf_st_new_arr, log_pi_arr, pre_log_pi_arr, params)

    """
    Gradient of Q functions:
    """
    ĝ_qf1 = ∇_Qf(Qf1, traj, Vf_tar_st2_arr, in_q_arr, params)
    ĝ_qf2 = ∇_Qf(Qf2, traj, Vf_tar_st2_arr, in_q_arr, params)

    """
    Gradient of Policy function: Select classic of maximum entropy
    """
    #ĝ_pl = ∇_Pl_SAC(Pl, traj, Qf1_st_new_arr, Vf_st_arr, rand_arr, log_pi_arr, pre_log_pi_arr, params)
    ĝ_pl = ∇_Pl(Pl, traj, Qf1_st_new_arr, Vf_st_arr, rand_arr, log_pi_arr, pre_log_pi_arr, params)


    return ĝ_vf, ĝ_qf1, ĝ_qf2, ĝ_pl
end

#Functions to Calculate the Value function loss + gradient
@everywhere function V_loss(Vf::Any, traj::Trajectory, Qf_st_new_arr::AbstractVector,
    log_pi_arr::AbstractVector, pre_log_pi_arr::AbstractVector, params::Params)

    loss = 0
    for i_t = 1: traj.len
        s_t = traj.s_t[i_t,:]
        V_t = mlp_nlayer_value_norm(Vf, s_t,params)[1]

        #Gather the V-function loss (zeros for classic approach, ones for maximum entropy)
        #loss += (V_t - Qf_st_new_arr[i_t] + log_pi_arr[i_t] * 1 - pre_log_pi_arr[i_t] * 1)^2
        loss += (V_t - Qf_st_new_arr[i_t])^2
    end

    return loss
end
@everywhere ∇_Vf = grad(V_loss,1)


#Functions to Calculate the Q function loss + gradient
@everywhere function Q_loss(Qf::Any, traj::Trajectory, Vf_tar_st2_arr::AbstractVector,
    in_q_arr::Array{Float64,2}, params::Params)

    loss = 0
    γ = params.γ_disc
    for i_t = 1: traj.len
        in_q = in_q_arr[i_t,:]
        r_t = traj.r_t[i_t]
        Q_t = mlp_nlayer_value_norm(Qf,in_q,params)[1]

        #Gather the Q-function loss:
        loss += (Q_t - r_t - γ*Vf_tar_st2_arr[i_t])^2
    end
    return loss
end
@everywhere ∇_Qf = grad(Q_loss,1)


#Functions to Calculate the Policy function loss + gradient
@everywhere function P_loss_SAC(Pl::Any, traj::Trajectory, Qf1_st_new_arr::AbstractVector,
    Vf_st_arr::AbstractVector, rand_arr::Array{Float64,2}, log_pi_arr::Array{Float64,1},
    pre_log_pi_arr::AbstractVector, params::Params)

    fac_mu  = 0
    fac_std = 1e-3
    fac_tan = 0
    u_range = params.u_range

    loss = 0

    for i_t = 1: traj.len
        #Take from struct
        s_t      = traj.s_t[i_t,:]
        rand_vec = rand_arr[i_t,:]

        #Regenerate to get gradient
        out_t, std      = mlp_nlayer_policy_SAC(Pl,s_t)
        sample          = out_t .+ rand_vec .* std
        log_pi          = sum(logpdf_SAC(sample,out_t,std))

        #The main Policy loss:
        policy_loss =  log_pi * (log_pi_arr[i_t]*1 - Qf1_st_new_arr[i_t] +  Vf_st_arr[i_t] - pre_log_pi_arr[i_t]*1)

        #Regression losses:
        mean_reg_loss = mean(fac_mu  * (out_t  .* out_t ))
        std_reg_loss  = mean(fac_std * (std    .* std   ))
        tan_reg_loss  = mean(fac_tan * (sample .* sample))

        #Policy Regression loss:
        policy_reg_loss = mean_reg_loss + std_reg_loss + tan_reg_loss

        #The final policy loss:
        loss += policy_loss + policy_reg_loss
    end

    return loss
end
@everywhere ∇_Pl_SAC = grad(P_loss_SAC,1)


#Functions to Calculate the Policy function loss + gradient
@everywhere function P_loss(Pl::Any, traj::Trajectory, Qf1_st_new_arr::AbstractVector,
    Vf_st_arr::AbstractVector, rand_arr::Array{Float64,2}, log_pi_arr::Array{Float64,1},
    pre_log_pi_arr::AbstractVector, params::Params)

    σ = params.σ

    #Initialize loss
    loss = 0

    for i_t = 1: traj.len
        #Take from struct
        s_t      = traj.s_t[i_t,:]
        rand_vec = rand_arr[i_t,:]

        #Regenerate to get gradient
        Pl_st           = mlp_nlayer_policy_norm(Pl,s_t,params)
        a_t_new         = Pl_st .+ (rand_vec .* σ)
        #p_a_t           = get_π_prob(Pl_st, σ, a_t_new)
        p_a_t           = (1 ./ (σ.*sqrt(2*pi))) .* exp.(((Pl_st .- a_t_new) .* (Pl_st .- a_t_new)) ./ (-2 .* σ .* σ))
        log_pi          = sum(log.(p_a_t))

        #The main Policy loss:
        policy_loss =  log_pi * (Qf1_st_new_arr[i_t] -  Vf_st_arr[i_t])

        #The final policy loss:
        loss += policy_loss
    end

    return loss
end
@everywhere ∇_Pl = grad(P_loss,1)
