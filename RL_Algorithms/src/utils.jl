using Dates
using JLD2

#Functions connected to the used Algorithm, but not directly to other domains:
#-------------------------------------------------------------------------------
#Return the L2Reg norm of a vector:
@everywhere L2Reg(x) = mean(x .* x)

#Return the Generalized Advantage Estimation:
@everywhere function td_lambda!(params::Params,traj::Trajectory)
    l_traj = traj.len
    γ      = params.γ_disc
    λ      = params.λ_critic

    Val = traj.val_t
    r_t = traj.r_t

    #Calculate the local TD-error:
    delta_V_i = zeros(l_traj)
    for i_t = 1:l_traj-1
        delta_V_i[i_t] = -Val[i_t] + r_t[i_t] + γ*Val[i_t+1]
    end

    #Discount the TD-Error as the Advantage
    R_t = vec(discount(delta_V_i,λ*γ))

    #Write result in trajectory
    traj.R_t = R_t
end


#Discount the Rewards
@everywhere function discount(rewards, fac)
    discounted = zeros(Float64, length(rewards), 1)
    discounted[end] = rewards[end]

    for i=(length(rewards)-1):-1:1
        discounted[i] = rewards[i] + fac * discounted[i+1]
    end
    return discounted
end


#Better: Do it after the episode was built:
function loss_a3c_mod(θ::Any,params::Params,traj::Trajectory)
    #Take params from struct:
    l_traj = params.l_traj
    γ      = params.γ_disc
    λ      = params.λ_actor

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
        A_t            = d_val + (γ*λ)*A_t                 #GAE update
        Err_actor     += logpdf(a_t,out_t[1],params) * A_t #Actor correction
        Err_critic    += 0.5 * (A_t * A_t)                 #Critic correction
    end

    #Return the final Error_Sum
    return Err_critic / l_traj
end

#Function to normalize the inputs of the NN => Chan 1982
@everywhere function update_variance!(rms1::RunMeanStd,rms2::RunMeanStd)
    #Update the mean
    δ = rms1.mean - rms2.mean
    new_mean = (rms1.count .* rms1.mean .- rms2.count .* rms2.mean) ./ (rms1.count + rms2.count)

    #Update the Variance
    m_1 = rms1.var .* (rms1.count - 1)
    m_2 = rms2.var .* (rms2.count - 1)
    M = m_1 .+ m_2 .+ (δ.*δ) .* rms1.count .* rms2.count ./ (rms1.count .+ rms2.count)
    new_count    = rms1.count + rms2.count
    new_variance = M ./ (new_count -1)
    new_std      = sqrt.(new_variance)

    #write new data into rms1:
    rms1.count = min(new_count,1e8) #limit to avoid static behaviour
    rms1.mean  = new_mean
    rms1.var   = new_variance
    rms1.std   = new_std
end

#Function to simply normalize the reward according to the current RMS
@everywhere function normalize_rms(rms::RunMeanStd, s_t)
    #Apply the obersvation-filter:
    return clamp.( (s_t .- rms.mean) ./ (rms.std .+ 1e-2), -5.0, 5.0)
end

#Clip_function
@everywhere function clip(x, lo::Real, hi::Real)
    if x < lo
        return lo
    elseif x > hi
        return hi
    else
        return x
    end
end

#https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm => Two-pass algorithm
#It avoids catastrophic cancellation
@everywhere function two_pass_variance!(all_traj::Traj_Data,rms::RunMeanStd, params::Params)
    #1. Init
    l_state  = length(params.s_init)
    n_traj   = params.n_traj
    x_mean   = zeros(l_state)
    x_sum    = zeros(l_state)
    x_sum_2  = zeros(l_state)
    n        = 0

    # Get the mean:
    for i_traj = 1:n_traj
        l_traj = all_traj.traj[i_traj].len
        x_sum += vec(sum(all_traj.traj[i_traj].s_t2[1:l_traj,:],dims = 1))
        n     += all_traj.traj[i_traj].len
    end
    x_mean = x_sum ./ n

    # Squared error from mean summed:
    for i_traj = 1:n_traj
        l_traj = all_traj.traj[i_traj].len
        for i_t = 1: l_traj
            s_t = all_traj.traj[i_traj].s_t2[i_t,:]
            x_sum_2 += (s_t .- x_mean) .* (s_t .- x_mean)
        end
    end

    #Calculate the final variance:
    variance = x_sum_2 ./ (n - 1)

    #Write data in struct
    rms.count = n
    rms.mean  = x_mean
    rms.var   = variance
    rms.std   = sqrt.(variance)
end









#Build minibatches from attained data:
#Idea Construct Batch_data as partial Trajectories!
@everywhere function build_minibatches(all_traj::Traj_Data,params::Params)
    batch_size = params.batch_size

    timesteps = 0
    #Build current number of timesteps for this iteration
    for i_traj = 1:params.n_traj
        timesteps += all_traj.traj[i_traj].len
    end
    all_traj.timesteps = timesteps

    #Determine number of minibatches:
    number_batches = Int(floor(timesteps / batch_size))

    #Now Initialize new Traj_Data containing only minibatches
    #---------------------------------------------------
    mini_batch = Traj_Data()
    mini_batch.traj = Array{Trajectory,1}(undef,number_batches)
    mini_batch.timesteps = number_batches*batch_size
    mini_batch.n_traj = number_batches


    #Create instance of trajectory:
    traj = init_traj(params,batch_size)

    #Fill the big struct with the small one:
    for i_traj = 1:number_batches
        mini_batch.traj[i_traj] = copy_traj(traj,params)
    end
    #----------------------------------------------------

    #Now fill the minibatch data struct with the data contained from the Trajectories:
    i_traj = 1
    i_tt = 1
    org_len = all_traj.traj[1].len

    for i_batch = 1:number_batches
        mini_batch.traj[i_batch].len = batch_size
        for i_tb = 1:batch_size

            #Copy the data
            copy_to_mini_batch!(mini_batch, all_traj, i_traj, i_batch, i_tt, i_tb)

            #Check if u reached the end of the current trajectory:
            if i_tt == org_len
                i_traj += 1 #switch to new trajectory
                i_tt = 0    #reset the timestep of the trajectory

                #Reset original Trajectory length
                if i_traj <= params.n_traj #avoid undefined step
                    org_len = all_traj.traj[i_traj].len
                end
            end

            #Increase timesteps of original trajectory:
            i_tt += 1
        end
    end

    #Return the mini_batch data struct
    return mini_batch
end



#Copy the data locally to the Minibatch
@everywhere function copy_to_mini_batch!(mini_batch::Traj_Data, all_traj::Traj_Data, i_traj::Int, i_batch::Int, i_tt::Int, i_tb::Int)
    #Copy states
    mini_batch.traj[i_batch].s_t[i_tb,:] = all_traj.traj[i_traj].s_t[i_tt,:]

    #Copy new states
    mini_batch.traj[i_batch].s_t2[i_tb,:] = all_traj.traj[i_traj].s_t2[i_tt,:]

    #Copy actions
    mini_batch.traj[i_batch].a_t[i_tb,:] = all_traj.traj[i_traj].a_t[i_tt,:]

    #Copy probability of actions:
    mini_batch.traj[i_batch].p_a_t[i_tb,:] = all_traj.traj[i_traj].p_a_t[i_tt,:]

    #Copy reward
    mini_batch.traj[i_batch].r_t[i_tb] = all_traj.traj[i_traj].r_t[i_tt]

    #Copy Value Result
    mini_batch.traj[i_batch].val_t[i_tb,:] = all_traj.traj[i_traj].val_t[i_tt,:]

    #Copy Output of the Network
    mini_batch.traj[i_batch].out_t[i_tb,:] = all_traj.traj[i_traj].out_t[i_tt,:]

    #Copy local Advantage
    mini_batch.traj[i_batch].A_t[i_tb] = all_traj.traj[i_traj].A_t[i_tt]

    #Copy local discounted Reward
    mini_batch.traj[i_batch].R_t[i_tb] = all_traj.traj[i_traj].R_t[i_tt]
end

#Save the data for analysis
function save_data(res::Array{Float64,1}, prec::Array{Float64,1}, θa::Any, θv::Any, all_traj::Traj_Data, i_iter::Int)
    filename = string("Results/Actor_Critic_",Dates.today(),"iteartion_",i_iter,".jld2")
    save(filename, "res", res, "prec", prec, "θa", θa, "θv", θv)
end

#This function pumps the new data into the big replay ring-buffer
function pump_buffer!(buffer::Traj_Data,all_traj::Traj_Data, params::Params)
    current_buffer_size = buffer.n_traj
    new_data_size       = all_traj.n_traj
    max_size            = params.buffer_size

    #If buffer not full simply fill:
    if current_buffer_size + new_data_size < max_size
        buffer.n_traj = current_buffer_size + new_data_size
        append!(buffer.traj,all_traj.traj)
        buffer.n_traj = current_buffer_size + new_data_size

    #If buffer not full, but would run full
    elseif current_buffer_size < max_size
        #First calculate the required data to completely fill the buffer
        app_size = max_size - current_buffer_size
        #simply fill to have concluded the buffer
        append!(buffer.traj,all_traj.traj[1:app_size])
        #now get the size required for shifting
        shift_size = new_data_size - app_size
        #now shift the buffer to delete old samples
        buffer.traj[1:end-shift_size] = buffer.traj[shift_size+1:end]
        #fill the "free" space with the still unused free samples
        buffer.traj[end-shift_size+1:end] = all_traj.traj[app_size+1:end]
        #set new buffer size
        buffer.n_traj = max_size

        print("--Buffer is now full!: ", buffer.n_traj,"--")

    #If the buffer is full and the new minibatch is not larger shift
    elseif new_data_size < max_size
        shift_size = new_data_size
        #now shift the buffer to delete old samples
        buffer.traj[1:end-shift_size] = buffer.traj[shift_size+1:end]
        #Now fill the "free" space with the new samples
        buffer.traj[end-shift_size+1:end] = all_traj.traj
        #set new buffer size
        buffer.n_traj = max_size

    #If the new buffer is larger than the oberall buffer just fill partly
    else
        #fill the buffer with the partly new data
        buffer.traj = all_traj.traj[1:max_size]
        #set new buffer size
        buffer.n_traj = max_size
    end
    #set timestepsize of buffer
    buffer.timesteps = buffer.n_traj * params.batch_size
end


"""
Simply trashy function to get the baseline
"""
function simple_baseline!(all_traj::Traj_Data, params::Params)
    n_traj = all_traj.n_traj
    γ = params.γ_disc
    max_len = 0
    #Get R_t for all timesteps
    for i_traj = 1:n_traj
        len = all_traj.traj[i_traj].len
        #update max_len
        if len > max_len
            max_len = len
        end
        R_to = all_traj.traj[i_traj].r_t[1]
        all_traj.traj[i_traj].R_t[1] = R_to
        for i_t = 2:len
            r_t = all_traj.traj[i_traj].r_t[i_t]
            all_traj.traj[i_traj].R_t[i_t] = γ*R_to + r_t
        end
    end
    #Get mean R_t = b_t
    for i_t = 1:max_len
        i_count   = 0
        R_all     = 0
        for i_traj = 1:all_traj.n_traj
            len = all_traj.traj[i_traj].len
            #Count if traj is long enough
            if i_t <= len
                i_count += 1
                R_all += all_traj.traj[i_traj].R_t[i_t]
            end
        end
        b_t = R_all / i_count
        for i_traj = 1:all_traj.n_traj
            len = all_traj.traj[i_traj].len
            #Count if traj is long enough
            if i_t <= len
                R_loc = all_traj.traj[i_traj].R_t[i_t]
                A_t   = R_loc - b_t
            end
        end
    end
end
