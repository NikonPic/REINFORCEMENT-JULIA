#Contains functions to update the networks

#Simply iterate trough the dataset and update:
function do_sac_update(Pl::Any,Qf1::Any, Qf2::Any, Vf::Any,Vf_tar::Any,buffer::Traj_Data,params::Params, opt_para_pl, opt_para_qf1, opt_para_qf2, opt_para_vf)
    n_traj = buffer.n_traj
    shuffle_arr = shuffle(1:n_traj)
    τ = params.τ

    for i_true = 1:n_traj
        i_traj = shuffle_arr[i_true]
        traj = buffer.traj[i_traj]

        #Get the gradients:
        ĝ_vf, ĝ_qf1, ĝ_qf2, ĝ_pl = grad_all_networks(traj, Vf, Vf_tar, Qf1, Qf2, Pl, params)

        #Update the Networks:
        Knet.update!(Vf, ĝ_vf, opt_para_vf)
        Knet.update!(Qf1, ĝ_qf1, opt_para_qf1)
        Knet.update!(Qf2, ĝ_qf2, opt_para_qf2)
        Knet.update!(Pl, ĝ_pl, opt_para_pl)
        Vf_tar = τ .* Vf .+ (1-τ) .* Vf_tar
    end
end

#This function will update the networks in parallel:
function do_sac_update_parallel!(Pl::Any,Qf1::Any,Qf2::Any,Vf::Any,Vf_tar::Any,buffer::Traj_Data,params::Params,
     opt_para_pl, opt_para_qf1, opt_para_qf2, opt_para_vf)

    #Take all partial traj and shuffle their order!
    n_traj = buffer.n_traj
    shuffle_arr = shuffle(1:n_traj)
    τ = params.τ

    active_workers = workers()
    n_workers = length(active_workers)

    ĝ_pl  = copy(Pl)
    ĝ_qf1 = copy(Qf1)
    ĝ_qf2 = copy(Qf2)
    ĝ_vf  = copy(Vf)

    # Create Channels storing results and starting values
    input_channel  = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks and traj
    output_channel = RemoteChannel(()->Channel{Tuple}(length(workers())+1)) # will contain networks

    # Create problem on every worker
    begin
        for p in workers()
            remote_do(distributed_sac_update, p, params, input_channel, output_channel)
        end
    end

    # Start optimizations
    n_parallel = Int(ceil(n_traj / (n_workers * 20)))
    prog = Progress(n_parallel,1)

    for i_parallel = 1 : n_parallel

        #1. Send the work tasks:
        for i_w = 1 : n_workers
            i_true = ((i_parallel-1)*n_workers + i_w) #get true order
            if i_true <= n_traj
                i_traj = shuffle_arr[i_true] #shuffle order
                traj_local = buffer.traj[i_traj]
                @async put!(input_channel, (traj_local, Pl, Qf1, Qf2, Vf, Vf_tar))
            end
        end

        # 2. Take the results
        ĝ_pl  *= 0
        ĝ_qf1 *= 0
        ĝ_qf2 *= 0
        ĝ_vf  *= 0

        for i_w = 1: n_workers
            i_true = ((i_parallel-1)*n_workers + i_w)
            if i_true <= n_traj
                #Take from Channel
                gradients = take!(output_channel)
                #Reorder
                ĝ_vf  += gradients[1]
                ĝ_qf1 += gradients[2]
                ĝ_qf2 += gradients[3]
                ĝ_pl  += gradients[4]
            end
        end

        #Update the Networks
        Knet.update!(Vf,  ĝ_vf,  opt_para_vf)
        Knet.update!(Qf1, ĝ_qf1, opt_para_qf1)
        Knet.update!(Qf2, ĝ_qf2, opt_para_qf2)
        Knet.update!(Pl,  ĝ_pl,  opt_para_pl)
        Vf_tar = τ .* Vf .+ (1-τ) .* Vf_tar

        #Update Progress
        #ProgressMeter.update!(prog,i_parallel) #update progress bar
    end

    # Tell workers that its done
    NaN_mat = zeros(length(params.batch_size))
    fill!(NaN_mat,NaN)
    ending_traj = init_traj(params,params.batch_size)
    ending_traj.r_t = NaN_mat

    for j = 1 : n_workers
        @async put!(input_channel, (ending_traj, Pl, Qf1, Qf2, Vf, Vf_tar))
    end

    #Close Channels
    sleep(0.1)
    close(input_channel)
    close(output_channel)

    return nothing
end


#Function running on the worker
@everywhere function distributed_sac_update(params::Params, input_channel::RemoteChannel, output_channel::RemoteChannel)

   #TODO BREAK!!!!


   # Start loop where work is done
   while true
       # break loop if NaN state received
       try
           #Take the new Input
           input = take!(input_channel)

           #Reorder data:
           traj_local = input[1]
           Pl         = input[2]
           Qf1        = input[3]
           Qf2        = input[4]
           Vf         = input[5]
           Vf_tar     = input[6]

           #Check if u reached final
           if (sum(isnan, traj_local.r_t) > 0)
               GC.gc()
               break
           end

           #DO THE WORK:
           #-----------------------------------------------------------------------
           ĝ_vf, ĝ_qf1, ĝ_qf2, ĝ_pl = grad_all_networks(traj_local, Vf, Vf_tar, Qf1, Qf2, Pl, params)
           #-----------------------------------------------------------------------

           put!(output_channel, (ĝ_vf, ĝ_qf1, ĝ_qf2, ĝ_pl) )

       catch
           println("Some problem occured, worker $(myid()) exiting trajectory optimization function.")
           break
       end
   end

   return nothing
end
