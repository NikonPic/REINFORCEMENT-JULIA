################################################################################
#------------------------------ Init Functions ---------------------------------
################################################################################
#Init the Net
function init_θ_combined(net_s::Array{Int,1})
    x      = net_s[1]
    hidden = net_s[2:end-1]
    ysize  = net_s[end]

    w = []
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, zeros(y, 1))
        x = y
    end
    push!(w, xavier(ysize, x)) # Prob actions
    push!(w, zeros(ysize, 1))
    push!(w, xavier(1, x))     # Value function
    push!(w, zeros(1, 1))

    return w
end

#Init the Net
function init_θ_single(net_s::Array{Int,1})
    x      = net_s[1]
    hidden = net_s[2:end-1]
    ysize  = net_s[end]

    w = []
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, xavier(y, 1))
        x = y
    end
    push!(w, xavier(ysize, x)) # Prob actions
    push!(w, xavier(ysize, 1))

    return w
end



################################################################################
#------------------------------ Output Functions -------------------------------
################################################################################
#Evaluate the Network (Actions + Value Function)
@everywhere function mlp_nlayer(w::Any,x::AbstractVector,params::Params)
    σ       = params.σ
    u_range = params.u_range

    for i = 1:2:length(w)-4
        x = relu.(w[i] * x .+ w[i+1])
    end
    mu_act   = vec((1 .- σ) .* u_range.*tanh.(w[end-3] * x .+ w[end-2]))
    value    = (w[end-1] * x .+ w[end])[1]
    return mu_act, value
end


#Evaluate the Policy-Network
@everywhere function mlp_nlayer_policy(w::Any,x::AbstractVector,params::Params)
    σ       = params.σ
    u_range = params.u_range

    for i = 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
    end
    mu_act   = (1 .- σ) .* u_range.*tanh.(w[end-1] * x .+ w[end])
    return vec(mu_act)
end

#Evaluate the Policy-Network
@everywhere function mlp_nlayer_policy_skeleton(w::Any,x::AbstractVector,params::Params)
    σ       = params.σ
    u_range = params.u_range

    for i = 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
    end
    mu_act   = 0.5 .* (1 .- 1 .* σ) .* u_range.*tanh.(w[end-1] * x .+ w[end]) .+ 0.5 * u_range
    return vec(mu_act)
end


#Evaluate the Value-Network
@everywhere function mlp_nlayer_value(w::Any,x::AbstractVector,params::Params)
    for i = 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
    end
    value    = (w[end-1] * x .+ w[end])[1]
    return value
end


#Get the log(pi(a|s))
@everywhere function logpdf(a_t,out_t,params::Params)
    fac = Float64(-log(sqrt(2pi)))
    r   = (a_t - out_t) ./ params.σ
    return -r .* r .* Float64(0.5) - (log.(params.σ)) .+ fac
end


################################################################################
#---------------------------- Gradient Functions -------------------------------
################################################################################

#Idea 1: Do everything in one go -> But bad performance!
@everywhere function lossfunc_do_all_combined(θ::Any,params::Params)
    #generate from struct:
    l_traj = params.l_traj
    log_pi = Any[]
    r_t    = Float64[]
    a_t    = Float64[]
    val_t  = Any[]
    s_t1   = copy(params.s_init)
    totalr = 0

    #Sum over one Trajectory and then build the gradient:
    for i_t = 1: l_traj
        out_t, loc_val  = mlp_nlayer(θ,s_t1,params) #evaluate
        loc_a_t         = sample_action(out_t, params.σ)[1]
        s_t, loc_r_t    = environment_call(s_t1,loc_a_t,params)

        loc_log_pi = logpdf(loc_a_t,out_t[1],params)

        #Extend the running storage
        push!(val_t ,loc_val)
        push!(r_t   ,loc_r_t[1])
        push!(a_t   ,loc_a_t)
        push!(log_pi,loc_log_pi)
        totalr += loc_r_t

    end
    println(totalr)

    #Get the Accumulated Reward:
    r_t = discount(r_t,params.γ_disc)
    A_t = r_t - val_t
    A_t = reshape(A_t, 1, length(A_t))

    #Return the final Sum over one Trajectory
    sum(A_t .* hcat(log_pi)) + L2Reg(A_t)
end

@everywhere ĝ_traj = grad(lossfunc_do_all_combined,1) #Gradient construction



#Better: Do it after the episode was built:
function loss_a3c_combined(θ::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    γ      = params.γ_disc
    λ      = params.λ_critic

    #Init
    Err_actor  = 0
    Err_critic = 0
    s_t = traj.s_t[1,:]
    val_old    = 0;
    A_t        = 0;

    #Hope: improved peroformance by going backwards...
    for i_t = l_traj-1:-1:1
        #Take from Struct
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t]
        r_t = traj.r_t[i_t]

        out_t, val_new = mlp_nlayer(θ,s_t,params)          #Evaluate the Network
        d_val          = r_t + γ*val_old - val_new         #Get local TD-error
        val_old        = val_new                           #Update old error
        A_t            = d_val + (γ*λ)*A_t                 #GAE update -> Advantage
        Err_actor     += sum(logpdf(a_t,out_t,params)*A_t) #Actor correction
        Err_critic    += 0.5*(A_t * A_t)                   #Critic correction
    end

    #Return the final Error_Sum
    return Err_actor + (Err_critic / l_traj)
end

a3cgrad_combined = grad(loss_a3c_combined,1) #Gradient construction


#Decompose into separate Value und Policy Function:
#-------------------------------------------------------------------------------

#Better: Do it after the episode was built:
@everywhere function loss_policy(θa::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len

    #Init
    Err_actor  = 0
    s_t = traj.s_t[1,:]

    #Hope: improved peroformance by going backwards...
    for i_t = 1:l_traj
        #Take from Struct
        s_t = traj.s_t[i_t,:]
        a_t = traj.a_t[i_t,:]
        r_t = traj.r_t[i_t]
        A_t = traj.A_t[i_t]

        #Only for Segway
        s_mod = s_t
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_t[1:2]
        s_mod[3]     = sin(s_t[3])
        s_mod[4]     = cos(s_t[3])
        s_mod[5:end] = s_t[4:end]
        #-------------------------------=#

        out_t = mlp_nlayer_policy_norm(θa,s_mod,params)  #.+ 0.5 #Evaluate the Network
        Err_actor += sum(logpdf(a_t,out_t,params) * A_t)  #Actor correction
    end

    #Return the final Error_Sum
    return Err_actor
end

@everywhere grad_policy = grad(loss_policy,1) #Gradient construction


#Better: Do it after the episode was built:
@everywhere function loss_value(θv::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    γ      = params.γ_disc
    λ      = params.λ_critic

    #Init
    Err_critic = 0
    s_t = traj.s_t[1,:]
    V_t_true   = 0;
    e_t        = 0;

    #Hope: improved peroformance by going backwards...
    for i_t = l_traj-1:-1:1
        #Take from Struct
        s_t = traj.s_t[i_t,:]
        r_t = traj.r_t[i_t]
        val_t2 = traj.val_t[i_t+1]

        #Only for Segway
        s_mod = s_t
        #=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_t[1:2]
        s_mod[3]     = sin(s_t[3])
        s_mod[4]     = cos(s_t[3])
        s_mod[5:end] = s_t[4:end]
        #-------------------------------=#

        V_t_est     = mlp_nlayer_value_norm(θv,s_mod,params)[1] #Estimate the Value
        V_t_true    = traj.R_t[i_t]                      #Get the true Value (TD1)
        #V_t_true    = r_t + γ*val_t2                     #Get the Bellman Error (TD0)
        e_t         = V_t_est - V_t_true                 #Calculate the error
        Err_critic += (e_t * e_t)                        #Collect the error
    end

    #Return the final Error_Sum
    return Err_critic / l_traj
end

@everywhere grad_value = grad(loss_value,1) #Gradient construction



#Better: Do it after the episode was built:
@everywhere function loss_value_bs(θv::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = traj.len
    γ      = params.γ_disc
    λ      = params.λ_critic

    #Init
    Err_critic = 0
    s_t = traj.s_t[1,:]
    V_t_true   = 0;
    e_t        = 0;

    #Hope: improved peroformance by going backwards...
    for i_t = l_traj-1:-1:1
        #Take from Struct
        s_t   = traj.s_t[i_t,:]
        r_t   = traj.r_t[i_t]
        val_t = traj.val_t[i_t]

        V_t_est     = mlp_nlayer_value(θv,s_t,params)[1] #Estimate the Value
        e_t         = V_t_est - r_t - γ * traj.val_t[i_t+1]
        Err_critic += (e_t * e_t)                        #Collect the error

        #Update eligibility
        #V_t_true    = (1-λ) * val_t + λ * (r_t + γ * V_t_true) #Get the true Value - bootstrapped
    end

    #Return the final Error_Sum
    return Err_critic / l_traj
end

@everywhere grad_value_bs = grad(loss_value_bs,1) #Gradient construction


function supervised_training!(all_traj::Traj_Data,θv::Array{Any,1}, params::Params,opt_para_v)
    test_num = Int(floor(0.7 * all_traj.n_traj))
    test_err_old = Inf
    test_err = 1e10
    ĝ_v = copy(θv)
    iter = 0
    println()
    println("----start-training-------")
    while test_err_old > test_err
        test_err_old = test_err
        test_err = 0
        iter += 1

        #Do a Training Batch:
        for i_train = 1: test_num
            ĝ_v = grad_value(θv,params,all_traj.traj[i_train])
            Knet.update!(θv, ĝ_v, opt_para_v)
        end

        #Evaluate the Error
        for i_test = test_num+1:all_traj.n_traj
            test_err += loss_value(θv,params,all_traj.traj[i_test])
        end

        if (((test_err_old/ test_err)^2) < 1.05) && iter > 3
            break
        end

        println(test_err)
    end
    println("------end-training-------")
end



################################################################################
#------------------------------ Util Functions ---------------------------------
################################################################################

#Filter NaN
@everywhere function filterNet_NaN!(θ::Any)
    for i_layer = 1: length(θ)
        for i_element = 1: length(θ[i_layer])
             if isnan(θ[i_layer][i_element])
                 θ[i_layer][i_element] = 0
             end
         end
     end
end


#Build the Multiplikation of the Neural Network
@everywhere function mult_net(θ1::Any,θ2::Any)
    mult_θ = similar(θ1)
    for i_layer = 1:length(θ1)
        mult_θ[i_layer] = θ1[i_layer] .* θ2[i_layer]
    end
    return mult_θ
end


#Build the Divison of two Nets
@everywhere function division_net(θ1::Any,θ2::Any)
    div_θ = similar(θ1)
    for i_layer = 1:length(θ1)
        div_θ[i_layer] = θ1[i_layer] ./ θ2[i_layer]
    end
    return div_θ
end


#create net to vector
@everywhere function net_to_vec(θ::Any)
    net_vec =  Array{Float64}(undef,0)
    for i_layer = 1:length(θ)
        append!(net_vec,vec(θ[i_layer]))
    end
    return net_vec
end


#create vector to net
@everywhere function vec_to_net(net_vec::Array{Float64,1}, θ::Any)
    vec_net = copy(θ) * 0
    global_count = 1
    for i_layer = 1:length(θ)
        for i_loc = 1:length(vec(θ[i_layer]))
            vec_net[i_layer][i_loc] = net_vec[global_count]
            global_count += 1
        end
    end
    return vec_net
end


#"""
#Return the Probability of a_t given the output of the network and action a_t
#"""
@everywhere function get_π_prob(net_out::Any,σ::Array{Float64,1},a_t::Array{Float64,1})
    return (1 ./ (σ.*sqrt(2*pi))) .* exp.(((net_out .- a_t) .* (net_out .- a_t)) ./ (-2 .* σ .* σ))
end
