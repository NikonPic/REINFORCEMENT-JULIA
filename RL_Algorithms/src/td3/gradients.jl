#containing functions trying to applicate the "Twin Delayed Deep Deterministic Policy Gradient" from https://arxiv.org/pdf/1802.09477.pdf
#First we need a policy with a variable gaussian distribution:
@everywhere using Statistics

@everywhere function grad_all_networks(pl_update::Bool,traj::Trajectory, Qf_1::Any, Qf_tar_1::Any, Qf_2::Any, Qf_tar_2::Any, Pl::Any, Pl_tar::Any, params::Params)
    """
    This function determines the gradients for all networks over a Trajectory
    """
    #Take from struct
    len     = traj.len
    a_dim   = length(params.u_range)
    s_dim   = length(params.s_init)
    γ       = params.γ_disc
    σ_pl    = params.σ_off_policy
    clip_pl = params.clip_σ

    #Preallocate target data:
    q_data         = zeros(len,s_dim+a_dim)
    q_target_1     = zeros(len)
    q_target_2     = zeros(len)

    """
    Precalculation to generate targets for the loss functions:
    This has to be done with the latest versions of networks
    """
    for i_t = 1: traj.len
        #Take from struct
        s_t           = traj.s_t[i_t,:] #state
        s_t2          = traj.s_t2[i_t,:] #new state
        a_t           = traj.a_t[i_t,:] #old actions
        r_t           = traj.r_t[i_t] #received reward
        q_data[i_t,:] = cat(s_t,a_t; dims = 1) #input of q-function

        #Sample new action
        out_pl_tar    = mlp_nlayer_policy_norm(Pl_tar, s_t2, params)

        #Apply random clipped seed
        a_t_new       = clipped_action(out_pl_tar, σ_pl, clip_pl, params)

        #Evaluate target networks
        in_q_new      = cat(s_t, a_t_new; dims = 1)
        Qf1_st_new    = mlp_nlayer_value_norm(Qf_tar_1,in_q_new,params)[1]
        Qf2_st_new    = mlp_nlayer_value_norm(Qf_tar_2,in_q_new,params)[1]

        #Save target data in Arrays, clip the Q-functions to avoid over-estimation!
        q_target_1[i_t] = r_t + γ * max(Qf1_st_new, Qf2_st_new)# * 0.5
        q_target_2[i_t] = r_t + γ * Qf2_st_new
    end

    """
    Gradient of Q functions:
    """
    ĝ_qf1 = ∇_Qf(Qf_1, traj, q_data, q_target_1, params)
    ĝ_qf2 = ∇_Qf(Qf_2, traj, q_data, q_target_1, params)

    """
    Gradient of Policy function:
    """
    if pl_update == true
        ĝ_pl = ∇_Pl(Pl, Qf_1, traj, params)
    else
        ĝ_pl = 0 # no update as q-function has to be learned first
    end

    #return the final gradients
    return ĝ_qf1, ĝ_qf2, ĝ_pl
end



#Functions to Calculate the Q function loss + gradient
@everywhere function Q_loss(Qf::Any, traj::Trajectory, q_data::Array{Float64,2}, q_target::Array{Float64,1}, params::Params)
    #initialize the loss
    loss = 0

    #Go over the whole trajectory
    for i_t = 1: traj.len

        #Evaluate the Q-function
        Q_t = mlp_nlayer_value_norm(Qf,q_data[i_t,:],params)[1]

        #Accumulate loss, by comparison with target:
        loss += (Q_t - q_target[i_t])^2
    end

    #return average loss:
    return loss / traj.len
end

@everywhere ∇_Qf = grad(Q_loss,1)



#Functions to Calculate the Policy function loss + gradient
@everywhere function Pl_loss(Pl,Qf_1,traj::Trajectory,params::Params)
    #general setup
    len = traj.len
    loss = 0

    #Over the whole minibatch:
    for i_t = 1:len
        #Take from struct and precalculate
        r_t = traj.r_t[i_t]
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t,:]

        #get inputs
        out_p  = mlp_nlayer_policy_norm(Pl,s_t,params)
        x      = cat(s_t,out_p; dims = 1) #stuff the input together

        #Here comes the evaluation as autograd has problems with calling the combined version
        #=----------------------------------------
        for i = 1:3:length(Qf_1)-2
            #1. Get the new summed input
            x = Qf_1[i] * x

            #2. Calculate the mean and std of the layer
            x_mean, x_std = get_norm(x)

            #3. Normalize activation by applying the norm and mean
            x_norm = (x .- x_mean) / x_std

            #4. Reparametrize the output
            x = x_norm .* Qf_1[i+1] + Qf_1[i+2]

            #5. Apply the non-linearity
            x = relu.(x)
        end
        out_q = (Qf_1[end-1] * x .+ Qf_1[end])[1]
        #-----------------------------------------=#

        #----------------------------------------
        for i = 1:2:length(Qf_1)-2
            x = relu.(Qf_1[i] * x .+ Qf_1[i+1])
        end
        out_q = (Qf_1[end-1] * x .+ Qf_1[end])[1]
        #-----------------------------------------=#

        loss  += out_q
    end

    #Return the final loss
    return (loss / len)
end

@everywhere ∇_Pl = grad(Pl_loss,1)



#Function adding random noise to taget policy action to avoid overfitting
@everywhere function clipped_action(out_pl_tar::AbstractVector, σ_pl::Float64, clip_pl::Float64, params::Params)
    #1. generate random vector
    rand_vec = randn(length(out_pl_tar))

    #2. scale on σ_pl and clip
    rand_vec = clamp.((rand_vec ./ σ_pl), -clip_pl, clip_pl)

    #3. match on output-space
    rand_vec = rand_vec .* params.u_range

    #4. add to deterministic output
    sample_out = out_pl_tar + rand_vec

    #5. return sampled output
    return sample_out
end
