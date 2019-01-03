@everywhere using LinearAlgebra
#TRPO Specifics:
#-------------------------------------------------------------------------------
#(1) Define Fisher-Matrix as Fisher-Vector-Product / Or calculate directly:
#-------------------------------------------------------------------------------

function Fisher_Vector_Product(all_traj::Traj_Data, θ, params::Params, v::Array{Float64,1})
    #Determine data to average from:
    n_traj = all_traj.n_traj
    shuffle_arr = shuffle(1:n_traj)
    l_traj = params.l_traj
    cg_damping = params.cg_damping
    data_size = Int(floor(n_traj*0.2)) + 1

    #Constants:
    FVP = net_to_vec(θ)*0 #Preallocation
    data_len = 0

    #Go over 25% of all collected data:
    for i_true = 1:data_size
        #shuffle the trajectory taken:
        i_traj = shuffle_arr[i_true]
        l_traj = all_traj.traj[i_traj].len
        data_len += l_traj

        for i_t = 1:l_traj
            #Take current state and build derivative
            s_t = all_traj.traj[i_traj].s_t[i_t,:] #Current state
            a_t = all_traj.traj[i_traj].a_t[i_t,:] #Current state
            Jt  = ∇logπ(θ,s_t,a_t,params)
            Jt_flat = net_to_vec(Jt)
            #Fisher Vector Product
            FVP += Jt_flat*dot(Jt_flat, v)
        end
    end

    FVP /= (data_len) #Average over collected data
    FVP += cg_damping * v      #Add the damping factor
    return FVP
end


#For small Networks: get all the Training data and collect the Fisher approximation!
function update_Fisher(all_traj::Traj_Data, θ, params::Params, Fisher_old)
    n_traj = all_traj.n_traj
    data_size = n_traj#Int(floor(n_traj/5)) + 1
    fac = 0.8 #Building the weighted averge of the FIM for increased performance!

    len = Int(sqrt(length(Fisher_old)))
    FIM = zeros(len,len)
    data_len = 0
    #Go over 10% of all collected data:
    for i_traj = 1:data_size
        l_traj = all_traj.traj[i_traj].len
        data_len += l_traj
        for i_t = 1:l_traj
            #Take current state and build derivative
            s_t = all_traj.traj[i_traj].s_t[i_t,:] #Current state
            a_t = all_traj.traj[i_traj].a_t[i_t,:]
            Jt  = ∇logπ(θ,s_t,a_t,params)
            Jt_flat = net_to_vec(Jt)

            #FIM += Jt_flat*D_KL_II*Jt_flat'
            Base.LinAlg.BLAS.syr!('U',1.0,Jt_flat,FIM)
        end
    end

    #Rebuild the whole Matrix from the Symmetric one:
    FIM = (Symmetric(FIM) / data_len) #Whole Matrix

    #Keep Running Average of FIM
    FIM = fac*FIM + (1-fac)*(Fisher_old)

    return FIM
end


#Build the logarithmic gradient
@everywhere function logπ(θ::Any, s_t::Array{Float64,1},a_t::Array{Float64,1},params::Params)
    #Only for segway:
    s_mod = s_t
    #=-----------
    s_mod = zeros(8)
    s_mod[1:2]   = s_t[1:2]
    s_mod[3]     = sin(s_t[3])
    s_mod[4]     = cos(s_t[3])
    s_mod[5:end] = s_t[4:end]
    #-----------=#

    out_t   = mlp_nlayer_policy_norm(θ,s_mod,params) #.+ 0.5#Evaluate the Network
    prob_pi = sum(logpdf(a_t,out_t,params))
    return prob_pi
end

@everywhere ∇logπ = grad(logπ,1)


#-------------------------------------------------------------------------------
#(2) Define Search direction F⁻¹g -> as result of Conjugate Gradient algorithm
#-------------------------------------------------------------------------------

#Conjugate Gradient Algorithm:
function conjugate_gradient(f_Ax::Function, b::Array{Float64,1}, all_traj::Traj_Data, θ, params::Params, cg_iters=10, residual_tol=1e-10)
    # Initialise
    d = copy(b)
    r = copy(b)
    x = copy(b)*0
    rTr = r'r
    #Do the Gradient Steps:
    for iter_conj = 1: cg_iters
        z = f_Ax(all_traj, θ, params, d) #Build Fisher_Vector_Product
        α = rTr / d'z
        x += α * d
        r -= α * z
        new_rTr = r'r
        β = new_rTr / rTr
        d = r + β * d
        rTr = new_rTr
        if rTr < residual_tol
            break
        end
    end
    return x #Is Inv(A)*g => NaturalGradient direction!
end


#Conjugate Gradient Algorithm:
function conjugate_gradient_with_Fisher(b::Array{Float64,1}, FIM, cg_iters=10, residual_tol=1e-10)
    # Initialise
    d = copy(b)
    r = copy(b)
    x = copy(b)*0
    rTr = r'r
    cg_damping = params.cg_damping

    #Do the Gradient Steps:
    for iter_conj = 1: cg_iters
        z = (FIM*d) + (cg_damping * d)   #Build Fisher_Vector_Product
        α = rTr / (d'z)
        x += α * d
        r -= α * z
        new_rTr = r'r
        β = new_rTr / rTr
        d = r + β * d
        rTr = new_rTr
        if rTr < residual_tol
            break
        end
    end
    return x #Is Inv(A)*g => NaturalGradient direction!
end


#-------------------------------------------------------------------------------
#(3) Perform line search along the new space to determine improvement
#-------------------------------------------------------------------------------
function linesearch(loss_all::Function,full_step, θa, all_traj::Traj_Data, params::Params)
    accept_ratio = .1
    max_backtracks = 10
    Loss_old = loss_all(θa,all_traj,params)
    stepfrac = 1.0

    for steps = 1: max_backtracks
        θ_new = θa - stepfrac * full_step
        Loss_new = loss_all(θ_new,all_traj,params)

        actual_improve = Loss_new - Loss_old

        #Break if u have improved:
        if actual_improve <= 0.0
            return θ_new
        end

        #Decrease the Stepsize
        stepfrac *= 0.6
    end

    return θa
end



#Calculate the loss on all trajectories / or at least a part of it..
function loss_all(θ,all_traj::Traj_Data,params::Params)
    n_traj  = all_traj.n_traj
    σ       = params.σ
    δ       = params.δ

    KL_mean  = 0
    L_theta  = 0
    data_len = 0

    #Size to be evaluating from:
    data_size = Int(floor(n_traj/2)) + 1 #here only on 10% of data...
    for i_traj = 1: data_size
        # As we regard costs => a negative Advantage is desired
        # Therefore we have a smaller loss is a negative advantage occurs
        l_traj = all_traj.traj[i_traj].len
        data_len += l_traj
        A_t = all_traj.traj[i_traj].A_t

        for i_t = 1: l_traj
            #Collect from struct
            s_t = all_traj.traj[i_traj].s_t[i_t,:]
            a_t = all_traj.traj[i_traj].a_t[i_t,:]
            #The two outputs:
            mu_old = all_traj.traj[i_traj].out_t[i_t,:]

            #Only for segway:
            s_mod = s_t
            #=-----------
            s_mod = zeros(8)
            s_mod[1:2]   = s_t[1:2]
            s_mod[3]     = sin(s_t[3])
            s_mod[4]     = cos(s_t[3])
            s_mod[5:end] = s_t[4:end]
            #-----------=#

            mu_new = mlp_nlayer_policy_norm(θ,s_mod,params)  #Evaluate the Network
            #The two probabilities
            p_old = prod(all_traj.traj[i_traj].p_a_t[i_t,:])
            p_new = prod(get_π_prob(mu_new,σ,a_t))
            #Update the losses
            KL_mean += p_new * log(p_new / p_old)#kl(mu_old,mu_new,σ)
            L_theta += (p_new / p_old) * A_t[i_t]
        end
    end
    #Average
    KL_mean /= (data_len)
    L_theta /= (data_len)

    #Check if the KL divergence is below δ
    if KL_mean < δ
        return L_theta
    end

    return Inf

end

#KL divergence between two gaussian distributions
function kl(μ1::Array{Float64,1},μ2::Array{Float64,1},σ::Array{Float64,1})
    return 0.5 * (μ1 - μ2)' * ((μ1 - μ2) .* (1 ./ (σ.*σ)))
end



#-------------------------------------------------------------------------------
# Performjm the complete TRPO update:
#-------------------------------------------------------------------------------

#Update without explicity calculating FIM
function do_TRPO_update(ĝ_a, all_traj::Traj_Data, θa, params::Params)
     # Flatten the gradient
     ĝ_v = net_to_vec(ĝ_a)

     #1. Determine the step direction with conjugate Gradient and FVP:
     stepdir = conjugate_gradient(Fisher_Vector_Product_parallel, ĝ_v, all_traj, θa, params)

     #2. Calculate the max step length β
     sTAs = stepdir' * Fisher_Vector_Product_parallel(all_traj, θa, params, stepdir)
     β = sqrt( (2*params.δ) / sTAs)
     full_step = vec_to_net(stepdir * β, θa)

     #3. Perform Line-Search along Step-direction until Constraint is fullfilled:
     θ_new = linesearch(loss_all, full_step, θa, all_traj, params)
     #θ_new = θa - full_step

     return θ_new
end


function do_TNPG_update(ĝ_a, all_traj::Traj_Data, θa, params::Params)
     # Flatten the gradient
     ĝ_v = net_to_vec(ĝ_a)

     #1. Determine the step direction with conjugate Gradient and FVP:
     stepdir = conjugate_gradient(Fisher_Vector_Product_parallel, ĝ_v, all_traj, θa, params)

     #2. Calculate the max step length β
     sTAs = stepdir' * Fisher_Vector_Product_parallel(all_traj, θa, params, stepdir)
     β = sqrt( (2*params.δ) / sTAs)
     full_step = vec_to_net(stepdir * β, θa)

     #3. Perform Line-Search along Step-direction until Constraint is fullfilled:
     θ_new = θa - full_step

     return θ_new
end


#Update with explicit calculation of FIM
function do_TRPO_update_small(ĝ, all_traj::Traj_Data, θ, params::Params, Fisher)
    ĝ_v = net_to_vec(ĝ)

    FIM = update_Fisher(all_traj, θ, params, Fisher)
    #@time FIM = update_Fisher_parallel(all_traj, θ, params, Fisher)

    stepdir = conjugate_gradient_with_Fisher(ĝ_v, FIM)

    sTAs = stepdir' * FIM * stepdir
    β = 0.9*sqrt( (2*params.δ) / sTAs)

    full_step = vec_to_net(stepdir * β, θ)

    θ_new = full_step#linesearch(loss_all, full_step, θ, all_traj, params)

    return θ_new

end

"""
Function for the parallel-update of the FIM
"""
function Fisher_Vector_Product_parallel(all_traj::Traj_Data, θ::Any, params::Params, v::Array{Float64,1})
    data_percent = 0.33 #Take only partial data for increased performance
    n_traj = all_traj.n_traj
    shuffle_arr = shuffle(1:n_traj) #shuffle trajectories taken

    l_traj = params.l_traj
    cg_damping = params.cg_damping
    data_size = Int(floor(n_traj*data_percent)) + 1

    FVP = net_to_vec(θ)*0 #Preallocation
    data_len = 0

    # Create Channels storing results and starting valus
    traj_channel = RemoteChannel(()->Channel{Trajectory}(length(workers())+1)) # will contain s_init
    FVP_channel  = RemoteChannel(()->Channel{Array{Float64,1}}(length(workers())+1)) # will contain traj and gradients

    # Create problem on every worker
    begin
        for p in workers()
            remote_do(distributed_fisher_VP, p,  θ, params, traj_channel, FVP_channel, v)
        end
    end

    # Start optimizations
    for i_true = 1:data_size
        i_traj = shuffle_arr[i_true]
        data_len += all_traj.traj[i_traj].len
        loc_traj = all_traj.traj[i_traj]
        @async put!(traj_channel,loc_traj)
    end

    # Collect the resulting data
    for i_true = 1:data_size
        #Take from Channel:
        fvp_loc = take!(FVP_channel)
        FVP += fvp_loc
    end


    # Tell workers that its done
    NaN_mat = zeros(length(params.l_traj))
    fill!(NaN_mat,NaN)
    ending_traj = init_traj(params,params.l_traj)
    ending_traj.r_t = NaN_mat

    for j = 1:length(workers())
            @async put!(traj_channel, ending_traj)
    end

    #Close Channels
    sleep(0.1)
    close(traj_channel)
    close(FVP_channel)

    #Get the final Fisher result
    FVP /= data_len
    FVP  = FVP + cg_damping * v
    #Return the Gradients:
    return FVP
end


# Function running on the worker calculating the FIsher Matrix
@everywhere function distributed_fisher_VP(θ::Any, params::Params ,traj_channel::RemoteChannel, FVP_channel::RemoteChannel, v::Array{Float64,1})
    loc_traj = init_traj(params,params.l_traj)
    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            loc_traj = copy_traj(take!(traj_channel),params)
            if (sum(isnan, loc_traj.r_t) > 0)
                break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #execute the calculation
        loc_FVP = local_Fisher_VP(θ,loc_traj,params,v)

        put!(FVP_channel, loc_FVP)
    end

    return nothing
end

#calculate the local Fisher-Matrix
@everywhere function local_Fisher_VP(θ::Any,loc_traj::Trajectory,params::Params, v::Array{Float64,1})
    FVP = copy(v)*0
    l_traj = loc_traj.len

    for i_t = 1:l_traj
        #Take current state and build derivative
        s_t = loc_traj.s_t[i_t,:] #Current state
        a_t = loc_traj.a_t[i_t,:]
        Jt  = ∇logπ(θ,s_t,a_t,params)
        Jt_flat = net_to_vec(Jt)
        #Fisher Vector Product
        FVP += Jt_flat*dot(Jt_flat, v)
    end

    return FVP
end

#calculate the local Fisher-Matrix- Version 2
@everywhere function local_Fisher_VP2(θ::Any,traj::Trajectory,params::Params, v::Array{Float64,1})

    Jt      = ∇logπ_traj(θ, traj, params)
    Jt_flat = net_to_vec(Jt)
    FVP     = Jt_flat*dot(Jt_flat, v)

    return FVP
end

#Build the logarithmic gradient
@everywhere function logπ_traj(θ::Any, traj::Trajectory, params::Params)
    l_traj  = traj.len
    prob_pi = 0

    for i_t = 1:l_traj
        s_t = traj.s_t[i_t,:] #Current state
        a_t = traj.a_t[i_t,:] #Current action

        out_t    = mlp_nlayer_policy_norm(θ,s_t,params) #Evaluate the Network
        prob_pi += sum(logpdf(a_t,out_t,params))
    end

    return prob_pi / l_traj
end

@everywhere ∇logπ_traj = grad(logπ_traj,1)


function kl_all(θ,all_traj::Traj_Data,params::Params)
    n_traj  = all_traj.n_traj
    σ       = params.σ
    data_len = 0

    KL_mean  = 0

    #Size to be evaluating from:
    data_size = Int(floor(n_traj/2)) + 1 #here only on 10% of data...
    for i_traj = 1: data_size
        # As we regard costs => a negative Advantage is desired
        # Therefore we have a smaller loss is a negative advantage occurs
        l_traj = all_traj.traj[i_traj].len
        data_len += l_traj
        A_t = all_traj.traj[i_traj].A_t

        for i_t = 1: l_traj
            #Collect from struct
            s_t = all_traj.traj[i_traj].s_t[i_t,:]
            a_t = all_traj.traj[i_traj].a_t[i_t,:]
            #The two outputs:
            mu_old = all_traj.traj[i_traj].out_t[i_t,:]

            #Only for segway:
            s_mod = s_t
            mu_new = mlp_nlayer_policy_norm(θ,s_mod,params)  #Evaluate the Network
            #The two probabilities
            p_old = prod(all_traj.traj[i_traj].p_a_t[i_t,:])
            p_new = prod(get_π_prob(mu_new,σ,a_t))
            #Update the losses
            #KL_mean += p_new * log(p_new / p_old)#kl(mu_old,mu_new,σ)
            KL_mean += kl(mu_old,mu_new,σ)
        end
    end
    #Average
    KL_mean /= (data_len)

    return KL_mean
end
