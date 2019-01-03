#Parallel Episodes
#-------------------------------------------------------------------------------
"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_episodes(params::Params,θa::Any,θv::Any,all_traj::Traj_Data,rms_state::RunMeanStd)
        n_traj = params.n_traj
        active_workers = workers()

        # Create Channels storing results and starting valus
        s_init_channel = RemoteChannel(()->Channel{Vector{Float64}}(length(workers())+1)) # will contain s_init
        result_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain traj and gradients

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_episodes, p, params, θa, θv, rms_state, s_init_channel, result_channel)
            end
        end

        # Start optimizations
        for i_traj = 1:n_traj
                s_init_local = params.s_init
                @async put!(s_init_channel, s_init_local)
        end

        # Collect the resulting data
        ĝ_a  = copy(θa)*0
        ĝ_v  = copy(θv)*0
        prec = 0
        res  = 0
        #prog = Progress(n_traj,1)
        for i_traj = 1:n_traj
            #Take from Channel:
            result = take!(result_channel)
            #Oder the Tuple:
            all_traj.traj[i_traj] = result[1]
            res  += result[4]
            prec += result[5]
        end
        ĝ_a /= n_traj
        ĝ_v /= n_traj

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
        return (res/n_traj, prec)
end



# Function running on the worker
@everywhere function distributed_episodes(params::Params, θa::Any, θv::Any, rms_state::RunMeanStd,
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
                #println("Worker $(myid()) received NaN start state, exiting trajectory optimization function.")
                break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #DO THE WORK:
        #-----------------------------------------------------------------------
        res = run_episode!(params,θa,θv,traj,rms_state)

        #Calculate the Advantage using GAE
        gae!(params,traj)

        #Discount the rewards for supervised Training of Value function
        traj.R_t = vec(discount(traj.r_t, params.γ_disc))

        #Calculate the Policy Gradient using GAE
        ĝ_a_loc = 0#grad_policy(θa,params,traj)

        #Calculate the Value Function gradient using TD(1)
        ĝ_v_loc = 0#grad_value_bs(θv,params,traj)

        #Collect Precision
        prec = loss_value(θv,params,traj)
        #-----------------------------------------------------------------------

        put!(result_channel, (traj, ĝ_a_loc, ĝ_v_loc, res, prec))
    end

    return nothing
end



#Parallel Episodes
#-------------------------------------------------------------------------------
"""
Function enabling parallel execution of the episodes by setting up the channels
"""
function parallel_episodes_combined(params::Params,θ::Any,all_traj::Traj_Data,rms_state::RunMeanStd)
        n_traj = params.n_traj
        active_workers = workers()

        # Create Channels storing results and starting valus
        s_init_channel = RemoteChannel(()->Channel{Vector{Float64}}(length(workers())+1)) # will contain s_init
        result_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain traj and gradients

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_episodes_combined, p, params, θ, rms_state, s_init_channel, result_channel)
            end
        end

        # Start optimizations
        for i_traj = 1:n_traj
                s_init_local = params.s_init
                @async put!(s_init_channel, s_init_local)
        end

        # Collect the resulting data
        ĝ = copy(θ)*0
        prec = 0
        res = 0
        prog = Progress(n_traj,1)
        for i_traj = 1:n_traj
                #Take from Channel:
                result = take!(result_channel)
                #Oder the Tuple:
                all_traj.traj[i_traj] = result[1]
                ĝ    += result[2]
                res  += result[3]
                prec += result[4]
                #update progress bar
                #ProgressMeter.update!(prog,i_traj)
        end
        ĝ /= n_traj

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
        return (ĝ, res/n_traj, prec/n_traj)
end


# Function running on the worker
@everywhere function distributed_episodes_combined(params::Params, θ::Any, rms_state::RunMeanStd,
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
                #println("Worker $(myid()) received NaN start state, exiting trajectory optimization function.")
                break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #DO THE WORK:
        #-----------------------------------------------------------------------
        res = run_episode!(params,θ,traj,rms_state)

        #Calculate the Advantage using GAE
        gae!(params,traj)

        #Discount the rewards for supervised Training of Value function
        traj.R_t = vec(discount(traj.r_t, params.γ_disc))

        #Calculate the Policy Gradient using GAE
        ĝ = 0#grad_policy(θa,params,traj)

        #Collect Precision
        prec = 0
        #-----------------------------------------------------------------------

        put!(result_channel, (traj, ĝ, res, prec))
    end

    return nothing
end








"""
Function enabling parallel execution of the gradient
"""
function parallel_gradient(params::Params,θa::Any,all_traj::Traj_Data)
        n_traj = all_traj.n_traj
        active_workers = workers()

        # Create Channels storing results and starting valus
        traj_channel = RemoteChannel(()->Channel{Trajectory}(length(workers())+1)) # will contain s_init
        result_channel = RemoteChannel(()->Channel{Array{Any,1}}(length(workers())+1)) # will contain traj and gradients

        # Create problem on every worker
        begin
            for p in workers()
                remote_do(distributed_gradient, p, params, θa, traj_channel, result_channel)
            end
        end

        # Start optimizations
        for i_traj = 1:n_traj
                traj_local = all_traj.traj[i_traj]
                @async put!(traj_channel, traj_local)
        end

        # Collect the resulting data
        ĝ_a = copy(θa)*0

        for i_traj = 1:n_traj
                #Take from Channel:
                ĝ_a += take!(result_channel)
        end
        ĝ_a /= n_traj

        NaN_mat = zeros(length(params.batch_size))
        fill!(NaN_mat,NaN)
        ending_traj = init_traj(params,params.batch_size)
        ending_traj.r_t = NaN_mat

        # Tell workers that its done
        for j = 1:length(workers())
                @async put!(traj_channel, ending_traj)
        end

        #Close Channels
        sleep(0.1)
        close(traj_channel)
        close(result_channel)

        #Return the Gradient:
        return ĝ_a
end


# Function running on the worker
@everywhere function distributed_gradient(params::Params, θa::Any, traj_channel::RemoteChannel, result_channel::RemoteChannel)

    # Preallocate trajectory
    traj_local = init_traj(params,params.l_traj)

    # Start loop where work is done
    while true
        # break loop if NaN state received
        try
            traj_local = take!(traj_channel)
            #Check if u reached final
            if (sum(isnan, traj_local.r_t) > 0)
                    #gc()
                    break
            end
        catch
            println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
            break
        end

        #-----------------------------------------------------------------------
        #Calculate the Policy Gradient using GAE
        ĝ_a_loc = grad_policy(θa,params,traj_local)
        #-----------------------------------------------------------------------

        put!(result_channel, ĝ_a_loc)
    end

    return nothing
end
