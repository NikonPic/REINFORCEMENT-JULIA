#some demos for the NIPS2018 Challenge: https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge
#PyCall needs to point to Anaconda3!
#trick: type "which python" in console, then reset Python_ENV
# ENV["PYTHON"] = "result of which-Python by anaconda 3"
#Pkg.build("PyCall")

#before statup of julia, type: "activate opensim-rl"

using PyCall
#@pyimport osim.env as osim
u_range = zeros(19) .+ 0.5
s_init  = zeros(160)

#u_range = zeros(18) .+ 0.5
#s_init  = zeros(160)

@everywhere function reset_Skeleton(env)
    return Float64.(env[:reset]())
end

#Interact with environment
@everywhere function environment_call_Skeleton(s_t1::Array{Float64,1},a_t::Array{Float64,1}, params::Params, env)

    #Ineract with environment:
    this_act = Float32.(a_t)

    observation, reward, done, info = env[:step](this_act)
    reward += s_t1[1]^2

    if done == true
        #punish for falling over
        reward  -= 100
        terminal = 1
    else
        terminal = 0
    end

    s_t2 = Float64.(observation)
    r_t  = Float64.(reward) * (-1)

    return s_t2 , r_t, terminal
end



@everywhere function run_episode_skeleton!(params::Params,θa::Any, θv::Any, traj::Trajectory, rms::RunMeanStd,env)
    #@pyimport osim.env as osim
    #env = osim.ProstheticsEnv(visualize=true)

    dt        = params.dt;
    t_horizon = params.t_horizon;
    s_init    = reset_Skeleton(env) #+ 0.1*sum(params.σ)*randn(length(params.s_init))
    l_traj    = params.l_traj;
    σ         = params.σ;
    u_range   = params.u_range;
    γ         = params.γ_disc;

    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0
    l_out = length(u_range)

    lim_s_t = 200

    for i_t = 1:l_traj
        #Clip to high values
        s_norm = s_t1

        #Evaluate the Network Output
        out_t  = mlp_nlayer_policy_norm(θa,s_norm,params) .+ 0.5
        val_t  = mlp_nlayer_value_norm(θv,s_norm,params)[1]

        #Sample:
        rand_vec = randn(l_out)

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ, rand_vec)
        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call_Skeleton(s_t1,loc_a_t,params,env)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t;
        traj.p_a_t[i_t,:]   = p_a_t;
        traj.r_t[i_t]       = loc_r_t*0.01;
        traj.val_t[i_t]     = val_t;
        traj.out_t[i_t,:]   = out_t;
        traj.len            = i_t
        totalr             += loc_r_t

        #Override the state:
        s_t1 .= s_t2;

        #Check if the terminal condition is fullfilled and break
        if terminal == 1
            traj.s_t[i_t,:] .= s_t1
            break
        end

    end

    return totalr / traj.len
end



@everywhere function test_env()
    try
        x = sum(env[:reset]())
        println(x)
        return x
    catch
        return 0
    end
end



"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_episodes_skeleton(params::Params,θa::Any,θv::Any,all_traj::Traj_Data,rms_state::RunMeanStd)
    n_traj = params.n_traj
    active_workers = workers()
    n_workers = length(active_workers)

    # Create Channels storing results and starting valus
    s_init_channel = RemoteChannel(()->Channel{Array{Float64,1}}(length(workers())+1)) # will contain s_init
    result_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain traj and gradients

    begin
        for p in workers()
            remote_do(distributed_episodes_skeleton, p, params, θa, θv, rms_state, s_init_channel, result_channel, p)
        end
    end


    # Start optimizations
    for i_traj = 1:n_traj
        s_init_local = params.s_init
        @async put!(s_init_channel, s_init_local)
    end

    prog = Progress(n_traj,1)
    res  = 0
    prec = 0

    ĝ_a = copy(θa)*0
    ĝ_v = copy(θv)*0

    for i_traj = 1:n_traj
        #Take from Channel:
        result = take!(result_channel)
        #Oder the Tuple:
        all_traj.traj[i_traj] = result[1]
        res  += result[4]
        prec += result[5]
        #update progress bar
        ProgressMeter.update!(prog,i_traj)
    end

    # Tell workers that its done
    for j = 1:length(workers())
        NaN_mat = zeros(length(params.s_init))
        fill!(NaN_mat,NaN)
        @async put!(s_init_channel, NaN_mat)
    end

    #Close Channels and clean rubbish
    sleep(0.1)
    close(s_init_channel)
    close(result_channel)

    #Return the final Results
    return  res, prec
end

# Function running on the worker
@everywhere function distributed_episodes_skeleton(params::Params, θa::Any, θv::Any, rms_state::RunMeanStd, s_init_channel::RemoteChannel, result_channel::RemoteChannel, p)

    #env = osim.ProstheticsEnv(visualize=true)

    # Preallocate trajectory
    traj = init_traj(params,params.l_traj)
    s_init_local = zeros(length(params.s_init))

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            #println("take envrionment")
            s_init_local = take!(s_init_channel)
            if (sum(isnan, s_init_local) > 0)
                #println("Done, garbage collection!")
                GC.gc()
                break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #DO THE WORK:
        #-----------------------------------------------------------------------
        res = run_episode_skeleton!(params,θa,θv,traj,rms_state,env)

        #Calculate the Advantage using GAE
        gae!(params,traj)

        #Discount the rewards for supervised Training of Value function
        traj.R_t = vec(discount(traj.r_t, params.γ_disc))

        #Calculate the Policy Gradient using GAE
        ĝ_a_loc = grad_policy(θa,params,traj)

        #Calculate the Value Function gradient using TD(1)
        ĝ_v_loc = 0#grad_value_bs(θv,params,traj)

        #Collect Precision
        prec = loss_value(θv,params,traj)
        #-----------------------------------------------------------------------

        put!(result_channel, (traj, ĝ_a_loc, ĝ_v_loc, res, prec))
    end

    return nothing
end


@everywhere function run_episode_skeleton_demo!(params::Params,θa::Any, θv::Any, traj::Trajectory, rms::RunMeanStd,env)
    #@pyimport osim.env as osim
    #env = osim.ProstheticsEnv(visualize=true)

    dt        = params.dt;
    t_horizon = params.t_horizon;
    s_init    = reset_Skeleton(env) #+ 0.1*sum(params.σ)*randn(length(params.s_init))
    l_traj    = params.l_traj;
    σ         = params.σ;
    u_range   = params.u_range;
    γ         = params.γ_disc;

    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0
    l_out = length(u_range)

    lim_s_t = 200

    for i_t = 1:l_traj
        #Take the unnormalizes state
        s_norm = s_t1

        #Evaluate the Network Output
        out_t  = mlp_nlayer_policy_norm(θa,s_norm,params) .+ 0.5
        val_t  = mlp_nlayer_value_norm(θv,s_norm,params)[1]

        rand_vec = zeros(l_out)

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ, rand_vec)
        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call_Skeleton(s_t1,loc_a_t,params,env)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_norm
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t;
        traj.p_a_t[i_t,:]   = p_a_t;
        traj.r_t[i_t]       = loc_r_t*0.01;
        traj.val_t[i_t]     = val_t;
        traj.out_t[i_t,:]   = out_t;
        traj.len            = i_t
        totalr             += loc_r_t

        #Override the state:
        s_t1 .= s_t2;

        #Check if the terminal condition is fullfilled and break
        if terminal == 1
            traj.s_t[i_t,:] .= s_t1
            break
        end

    end

    return totalr / traj.len
end
