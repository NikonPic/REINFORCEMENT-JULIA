#Video:
function make_video(all_traj::Traj_Data,params::Params, i_iter= 300::Int)
    i_best = 1
    max_len = 0

    #Preselect good trajectory
    for i_traj = 1: params.n_traj
        if  all_traj.traj[i_traj].len > max_len
            max_len = all_traj.traj[i_traj].len
            i_best =  i_traj
        end
    end

    #Take data for video
    t_plot = 0:params.dt:params.t_horizon;
    t_plot = t_plot[1:max_len]
    x1 = all_traj.traj[i_best].s_t[1:max_len,:]
    u1 = all_traj.traj[i_best].a_t[1:max_len,:]'
    filename = string("Results/Videos/Actor_Critic_",Dates.today(),"_iter_",i_iter,".mp4")

    #plot_all_segways(all_traj, filename, params; options = Dict(:U => u1, :size => (1920, 1080)))

    #Select the Environment to visualise here!
    visualise(:InvPendulum, x1', t_plot, filename; options = Dict(:U => u1, :size => (1920, 1080)))
end


#Visualization:
#-------------------------------------------------------------------------------
@everywhere function plot_stuff(params::Params, traj::Trajectory, a_t_old::Array{Float64,1},
    weights_old::Array{Float64,1}, i_iter::Int,θ::Array{Any,1},
    res::Array{Float64,1},precision::Array{Float64,1})
    #Plotting part
    #--------------------------------------------------
    figure(1)
    clf()
    subplot(311)
    a_t_new = traj.a_t
    plot(0:params.dt:params.t_horizon,a_t_old,label="a_old");
    plot(0:params.dt:params.t_horizon,a_t_new,label="a_new");
    #plot(t,uout,label="perfect");
    xlabel("t")
    ylabel("u_motor")
    legend()
    a_t_old = a_t_new


    subplot(312)
    lab = ("x","v","dφ","φ", )

    for i_pl = 1: length(traj.s_t[1,:])
        plot(0:params.dt:params.t_horizon,traj.s_t[:,i_pl],label=lab[i_pl]);
    end
    #plot(t,x,label="perfect");
    xlabel("t")
    ylabel("s_t")
    legend()


    subplot(313)
    weights = net_to_vec(θ)
    plot(weights_old[:,1],label="θ_old");
    plot(weights,label="θ_new")
    #legend()
    ylabel("Weights")


    figure(2)
    clf()
    subplot(211)
    plot(res[1:i_iter])
    xlabel("Iterations")
    ylabel("Cost")

    subplot(212)
    plot(precision[1:i_iter])
    xlabel("Iterations")
    ylabel("Advantage-Precision")

    sleep(0.5)

    return a_t_new ,weights
    #--------------------------------------------------
end



#-------------------------------------------------------------------------------
@everywhere function plot_stuff(params::Params, all_traj::Traj_Data, a_t_old::Array{Float64,2},
    weights_old::Array{Float64,2}, i_iter::Int,θa,θv,
    res::Array{Float64,1},precision::Array{Float64,1}, ĝ_a, ĝ_v)
    #Plotting part
    #--------------------------------------------------
    #select best trajectory:
    max_len = 0
    i_best = 1
    for i_traj = 1: params.n_traj
        if max_len < all_traj.traj[i_traj].len
            i_best = i_traj
            max_len = all_traj.traj[i_traj].len
        end
    end

    len = all_traj.traj[i_best].len
    ##################################

    figure("Trajectory Infos")
    clf()
    subplot(311)
    #plot(0:params.dt:params.t_horizon,a_t_old,label="a_old");
    a_t_new = all_traj.traj[i_best].a_t
    plot(0:params.dt:params.t_horizon,a_t_new,label="a_new");
    #plot(t,uout,label="perfect");
    xlabel("t")
    ylabel("u_motor")
    #legend()
    a_t_old = copy(a_t_new)


    subplot(312)
    lab = ("v","x","dφ","φ","a","b","c","d","e","f","g","h", "i")

    for i_pl = 1: length(all_traj.traj[i_best].s_t[1,:])
        plot(0:params.dt:params.t_horizon,all_traj.traj[i_best].s_t[:,i_pl])#,label=lab[i_pl]);
    end
    xlabel("t")
    ylabel("s_t")
    #legend()


    subplot(313)
    plot(weights_old[:,1],label="θ_old")
    weights_old[:,1] = net_to_vec(θa)
    plot(weights_old[:,1],label="θ_new")
    #legend()
    ylabel("Weights-Policy")

    ##################################
    figure("Improvement: Actor improvement")
    clf()
    #subplot(211)
    plot(res[1:i_iter])
    xlabel("Iterations")
    ylabel("Cost")


    figure("Improvement: Value improvement")
    clf()
    if i_iter < 11
        start = 1
    else
        start = i_iter - 10
    end
    plot(precision[1:i_iter])
    xlabel("Iterations")
    ylabel("Value-Precision")

    sleep(0.5)

    #Plot Advantage:
    ##################################
    figure("Advantage")
    clf()
    for i_traj = 1: params.n_traj
        len = all_traj.traj[i_traj].len
        plot(all_traj.traj[i_traj].A_t[1:len])
        ylabel("Advantage")
    end

    figure("Reward_sum")
    clf()
    rew_sum = zeros(params.l_traj)
    for i_t = 1:params.l_traj
        active_traj = 0

        for i_traj = 1:params.n_traj
            if all_traj.traj[i_traj].r_t[i_t] != 0
                active_traj += 1
                rew_sum[i_t] += all_traj.traj[i_traj].r_t[i_t]
            end
        end

        if active_traj > 0
            rew_sum[i_t] /= active_traj
        end
    end
    plot(rew_sum)

    #Plot Rewards:
    ##################################
    figure("Rewards")
    clf()
    for i_traj = 1: params.n_traj
        len = all_traj.traj[i_traj].len
        plot(all_traj.traj[i_traj].r_t[1:len])
        ylabel("Costs")
    end

    len = all_traj.traj[i_best].len
    #Plot Trajectory and Advantage, Value comparison
    ##################################
    figure("State Details")
    clf()
    state_len = length(all_traj.traj[i_best].s_t[1,:])
    #single comparison of one trajectory:
    R_t = discount(all_traj.traj[i_best].r_t[1:len],params.γ_disc)

    subplot(2,1,1)
    lab = ("x","y","φz_Ball","ϕ_y","ϕ_x","ϕ_z","δx","δy","δφz_Ball","δϕ_y","δϕ_x","δϕ_z", "i","j","k")

    for i_pl = 1: Int(floor(state_len/2))
        plot(all_traj.traj[i_best].s_t[1:len,i_pl])#,label=lab[i_pl]);
    end

    subplot(2,1,2)
    for i_pl = (Int(floor(state_len/2))+1) : state_len
        plot(all_traj.traj[i_best].s_t[1:len,i_pl])#,label=lab[i_pl]);
    end

    figure("Advantage and Value")
    clf()
    plot(all_traj.traj[i_best].val_t[1:len], label="Qf_1 (half- lr)")
    plot(all_traj.traj[i_best].A_t[1:len], label="Qf_2 (double- lr)")

    plot(R_t, label="R_t")
    legend()
    return a_t_old, weights_old
    #--------------------------------------------------
end

#3D Plot of the Advantage:
function polt_surface_single(params::Params, θv::Array{Any,1}, θa::Array{Any,1})

    n = 100
    x = linspace(-2, 2, n)
    y = linspace(-2, 2,n)

    xgrid = repmat(x',n,1)
    ygrid = repmat(y,1,n)

    values = zeros(n,n)
    out    = zeros(n,n)

    for i in 1:n
        for j in 1:n
            s_t = [x[i];y[j]]
            values[i:i,j:j] =  mlp_nlayer_value(θv,s_t,params)[1]
            out[i:i,j:j]    =  mlp_nlayer_policy(θa,s_t,params)[1]
        end
    end

    fig = figure("Advantage_Visulization_separated")
    clf()
    plot_surface(xgrid,ygrid,values)
    plot_surface(xgrid,ygrid,out)
    xlabel("Velocity")
    ylabel("Position")
    title("Value and Policy Output")
end


function polt_surface_combined(params::Params, θv::Array{Any,1})

    n = 100
    x = linspace(-0.07, 0.07, n)
    y = linspace(-2, 2,n)

    xgrid = repmat(x',n,1)
    ygrid = repmat(y,1,n)

    values = zeros(n,n)
    out    = zeros(n,n)

    for i in 1:n
        for j in 1:n
            s_t = [x[i];y[j]]
            out_s, values[i:i,j:j] =  mlp_nlayer(θv,s_t,params)
            out[i:i,j:j] = out_s[1]
        end
    end

    fig = figure("Advantage_Visulization_combined")
    clf()
    plot_surface(xgrid,ygrid,values,label ="Value")
    plot_surface(xgrid,ygrid,out,label="Action")
    xlabel("Velocity")
    ylabel("Position")
    title("Value Estimation")
end
