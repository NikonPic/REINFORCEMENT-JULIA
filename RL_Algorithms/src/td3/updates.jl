#Contains functions to update the networks

#Simply iterate trough the dataset and update:
function do_td3_update(Qf_1::Any, Qf_tar_1::Any, Qf2::Any, Qf_tar_2::Any, Pl::Any, Pl_tar::Any,
    buffer::Traj_Data,params::Params, opt_para_pl, opt_para_qf1, opt_para_qf2)

    n_traj = buffer.n_traj
    shuffle_arr = shuffle(1:n_traj)
    τ = params.τ

    #Parameter selecting when to update the target and policy networks:
    d = params.d_update

    for i_true = 2:n_traj
        i_traj = shuffle_arr[i_true]
        traj = buffer.traj[i_traj]

        #Only every d-steps: Delayed Policy Update
        if mod(i_true, d) == 0
            pl_update = true
        else
            pl_update = false
        end

        #Get the gradients:
        ĝ_qf1, ĝ_qf2, ĝ_pl = grad_all_networks(pl_update, traj, Qf_1, Qf_tar_1, Qf_2, Qf_tar_2, Pl, Pl_tar, params)

        #Always update the normal Q-functions
        Knet.update!(Qf_1, ĝ_qf1, opt_para_qf1)
        Knet.update!(Qf_2, ĝ_qf2, opt_para_qf2)

        #Do the delayed Policy update:
        if pl_update == true
            #update the policy function
            Knet.update!(Pl, ĝ_pl, opt_para_pl)

            #Update all Target functions
            Qf_tar_1 = τ .* Qf_1 .+ (1-τ) .* Qf_tar_1
            Qf_tar_2 = τ .* Qf_2 .+ (1-τ) .* Qf_tar_2
            Pl_tar   = τ .* Pl   .+ (1-τ) .* Pl_tar
        end
    end
end

#This function will update the networks in parallel:
function do_td3_update_parallel!(Qf_1::Any, Qf_tar_1::Any, Qf_2::Any, Qf_tar_2::Any, Pl::Any, Pl_tar::Any,
    buffer::Traj_Data,params::Params, opt_para_pl, opt_para_qf1, opt_para_qf2)

    """
    Setup the problem
    """
    #Take all partial traj and shuffle their order!
    n_traj = buffer.n_traj
    shuffle_arr = shuffle(1:n_traj)
    τ = params.τ

    active_workers = workers()
    n_workers = length(active_workers)

    ĝ_qf1 = copy(Qf_1)
    ĝ_qf2 = copy(Qf_2)
    ĝ_pl  = copy(Pl)

    # Create Channels storing results and starting values
    input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
    output_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks

    # Create problem on every worker
    begin
        for p in workers()
            remote_do(distributed_td3_update, p, params, input_channel, output_channel)
        end
    end

    """
    Do the work in parallel
    """
    partial_step = 0.1
    n_parallel   = Int(ceil((n_traj*partial_step) / n_workers))
    #prog       = Progress(n_parallel,1)

    #Parameter selecting when to update the target and policy networks:
    d = params.d_update

    for i_parallel = 1 : n_parallel

        #Only every d-steps: Delayed Policy Update
        if mod(i_parallel, d) == 0
            pl_update = true
        else
            pl_update = false
        end

        #1. Send the work tasks:
        for i_w = 1 : n_workers
            i_true = ((i_parallel-1)*n_workers + i_w) #get true order
            if i_true <= n_traj
                i_traj  = shuffle_arr[i_true] #shuffle order
                traj    = buffer.traj[i_traj]
                @async put!(input_channel, (pl_update, traj, Qf_1, Qf_tar_1, Qf_2, Qf_tar_2, Pl, Pl_tar))
            end
        end

        # 2. Take the results
        ĝ_qf1 *= 0
        ĝ_qf2 *= 0
        ĝ_pl  *= 0

        for i_w = 1: n_workers
            i_true = ((i_parallel-1)*n_workers + i_w)
            if i_true <= n_traj
                #Take from Channel
                gradients = take!(output_channel)
                #Reorder
                ĝ_qf1 += gradients[1]
                ĝ_qf2 += gradients[2]

                #Only when policy-gradient was calculated
                if pl_update == true
                    ĝ_pl  += gradients[3]
                end

            end
        end

        #Always update the normal Q-functions
        Knet.update!(Qf_1, ĝ_qf1, opt_para_qf1)
        Knet.update!(Qf_2, ĝ_qf2, opt_para_qf2)

        #Only frequently update target networks and Policy
        if pl_update == true
            #update the policy network
            Knet.update!(Pl, ĝ_pl, opt_para_pl)

            #Update all target networks
            Qf_tar_1 = τ .* Qf_1 .+ (1-τ) .* Qf_tar_1
            Qf_tar_2 = τ .* Qf_2 .+ (1-τ) .* Qf_tar_2
            Pl_tar   = τ .* Pl   .+ (1-τ) .* Pl_tar
        end

        #Update Progress
        #ProgressMeter.update!(prog,i_parallel) #update progress bar
    end

    """
    Close the channels
    """
    # Tell workers that its done
    NaN_mat = zeros(params.batch_size)
    fill!(NaN_mat,NaN)
    ending_traj = init_traj(params,params.batch_size)
    ending_traj.r_t = NaN_mat

    for j = 1 : n_workers
        @async put!(input_channel, (true, ending_traj, Qf_1, Qf_tar_1, Qf_2, Qf_tar_2, Pl, Pl_tar))
    end

    #Close Channels
    sleep(0.1)
    close(input_channel)
    close(output_channel)

    return nothing
end


#Function running on the worker
@everywhere function distributed_td3_update(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)
   # Start loop where work is done
   while true
       # break loop if NaN state received
       try
           #Take the new Input
           inputs = take!(input_channel)

           #Reorder data:
           pl_update  = inputs[1]
           traj       = inputs[2]
           Qf_1       = inputs[3]
           Qf_tar_1   = inputs[4]
           Qf_2       = inputs[5]
           Qf_tar_2   = inputs[6]
           Pl         = inputs[7]
           Pl_tar     = inputs[8]

           #Check if u reached final
           if (sum(isnan, traj.r_t) > 0)
               GC.gc()
               break
           end

           #DO THE WORK: (in ./td3/gradients)
           #-----------------------------------------------------------------------
           ĝ_qf1, ĝ_qf2, ĝ_pl = grad_all_networks(pl_update, traj, Qf_1, Qf_tar_1, Qf_2, Qf_tar_2, Pl, Pl_tar, params)
           #-----------------------------------------------------------------------

           if pl_update == true
               put!(output_channel, (ĝ_qf1, ĝ_qf2, ĝ_pl))
           else
               put!(output_channel, (ĝ_qf1, ĝ_qf2))
           end

       catch
           println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
           break
       end
   end

   return nothing
end
