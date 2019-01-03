using JLD2
using FileIO

include("src/init.jl")

"""
This file contains the script to evaluate the trained controller on the real physical system
"""


params.t_horizon = 300;       #Time until end
params.l_traj    = length(0:params.dt:params.t_horizon); #Total timesteps per episode
#params.σ *= 0.5
params.n_traj = 10;
rms_all = init_rms(params.s_init)





#function for smoothing the data
function f_filt(x)
    l = length(x)
    tau = 0.01
    y = zeros(l)
    y[1] = x[1]

    for i = 2:l
        y[i] = (1-tau)*y[i-1] + tau *x[i]
    end

    return y
end


"""
1. Load the trained network:
"""

#SELECT THE TRAINING RESULTS TO BE EVALUATED FROM:
#-------------------------------------------------------------------------------
trained_data = load("Results/10_01_real_segway/train_data.jld")
#trained_data = load("Results/Videos/08_30_PPO_Ball/Actor_Critic_2018-09-04iteartion_2000.jld")
#trained_data = load("Results/Videos/09_27_ballbot_more/Actor_Critic_2018-09-27iteartion_330.jld")
#trained_data = load("Results/Videos/09_26_ballbot/Actor_Critic_2018-09-26iteartion_1080.jld")

trained_data0 = load("Results/11_07_longterm_ball/1_all.jld2")
trained_data1 = load("Results/11_07_longterm_ball/PPO1_1_1500.jld2")
trained_data2 = load("Results/11_07_longterm_ball/PPO2_1_1500.jld2")

#Rewrite the results
data0 = trained_data0["result"][2,:]
data1 = trained_data1["(θa, θv, all_traj, best_net, result, kl_res)"][5][5,:]
data2 = trained_data2["(θa, θv, all_traj, best_net, result, kl_res)"][5][6,:]


θa  = trained_data1["(θa, θv, all_traj, best_net, result, kl_res)"][1]
θa2 = trained_data2["(θa, θv, all_traj, best_net, result, kl_res)"][1]


θa  = trained_data["θa"]
θa2 = trained_data["θa"]
#-------------------------------------------------------------------------------



# Test the network for functionality:
res = zeros(2)
prec = zeros(2)
i_iter = 1

all_traj  = init_all_traj(params)
traj = all_traj.traj[1]
#(ĝ_a, ĝ_v, res[i_iter], prec[i_iter]) = parallel_episodes(params,θa,θv,all_traj,rms_all)
a_t_old = all_traj.traj[1].a_t
weights_old = zeros(length(net_to_vec(θa)),2)

#a_t_old ,weights_old = plot_stuff(params, all_traj, a_t_old,weights_old, i_iter,θa, θv, res, prec/params.n_traj, ĝ_a, ĝ_v)


"""
2. we need a task-trajectory, that the segway has to follow:
"""

using HDF5

c = h5open("Results/10_01_real_segway/data.h5", "r") do file
    read(file, "xu")
end

h5file = h5open("Results/10_01_real_segway/trajectory.h5", "r")
des_traj = read(h5file)["des_traj"] .* 0.5
close(h5file)


using HDF5

c = h5open("Results/10_01_real_segway/data.h5", "r") do file
    read(file, "xu")
end

t_start = 10

x_real = Float64.(c)[1:7,:]
u1     = Float64.(c)[8:9,:]
t_real = Float64.(c)[10,:]
len_real = length(t_real)

des_traj_real = zeros(length(x_real[1,:]),2)
iter_beginn = t_start * 100
iter_end    = iter_beginn + length(des_traj[:,1])
l_end = len_real - iter_end +1
des_traj_real[iter_beginn:iter_end-1,:] = des_traj
des_traj_real[iter_end:end,:] = des_traj[end,:]' .* ones(l_end,2)


l_des = length(des_traj_real[:,1])

traj = init_traj(params,length(des_traj_real[:,1]))
traj2 = init_traj(params,length(des_traj_real[:,1]))


"""
3. The System was trained on going to the center starting from point (1,0) with random rotation.
    => we need to rotate the koordinate system at every step
"""
function run_episode_continous!(params::Params,θa::Any, traj::Trajectory, rms::RunMeanStd, des_traj::Array{Float64,2})
    dt          = params.dt
    t_horizon   = params.t_horizon
    s_init      = params.s_init
    σ           = params.σ
    u_range     = params.u_range
    l_out       = length(u_range)

    #Initialize :
    s_t1 = copy(s_init)
    s_t2 = similar(s_t1)
    totalr = 0
    rand_vec = zeros(l_out)

    #Declare the right position here:
    des_x      = des_traj[1,1]
    des_y      = des_traj[1,2]

    l_traj      = length(des_traj[:,1])
    l_state     = length(s_init)



    s_fake = s_init
    s_t1   = zeros(l_state)
    s_t    = copy(s_t1)


    for i_t = 1:l_traj

        #Update input
        s_t .= s_t1
        s_fake = s_t

        #Recalculate the Input
        d_x      = des_traj[i_t,1] - s_t1[1]
        d_y      = des_traj[i_t,2] - s_t2[2]


        scale = (d_x^2 + d_y^2)
        part  = 0.3

        s_fake[1]    = -sign(d_x) * sqrt(scale - d_y^2) * part
        s_fake[2]    = -sign(d_y) * sqrt(scale - d_x^2) * part

        #s_fake += 0.000*randn(length(s_fake))

        s_fake[1]    = -sign(d_x) * 1*min(abs(d_x),part)
        s_fake[2]    = -sign(d_y) * 1*min(abs(d_y),part)

        #-=---------------------------------
        s_mod        = zeros(8)
        s_mod[1:2]   = s_fake[1:2]
        s_mod[3]     = sin(s_fake[3])
        s_mod[4]     = cos(s_fake[3])
        s_mod[5:end] = s_fake[4:end]
        #-------------------------------=#


        #Evaluate the Network Output
        out_t = mlp_nlayer_policy(θa,s_mod,params)

        #Sample:
        rand_vec = randn(l_out) .* 0.0

        #Get Action with stochastic policy:
        loc_a_t = sample_action(out_t, params.σ, rand_vec)
        #loc_a_t = out_t
        p_a_t   = get_π_prob(out_t,σ,loc_a_t)

        act_fac = 3
        #Interact with the Environment:
        s_t2, loc_r_t, terminal = environment_call(s_t1,act_fac*loc_a_t,params)

        #Save data in trajectory:
        traj.s_t[i_t,:]    .= s_t1
        traj.s_t2[i_t,:]   .= s_t2
        traj.a_t[i_t,:]    .= loc_a_t
        traj.p_a_t[i_t,:]   = p_a_t
        traj.r_t[i_t]       = loc_r_t*0.01

        traj.out_t[i_t,:]   = out_t
        traj.len            = i_t
        totalr             += loc_r_t

        #Override the state:
        s_t1 .= s_t2;

        #Check if the terminal condition is fullfilled and break
        if terminal == 1
                break
        end

    end
    return totalr / traj.len
end

run_episode_continous!(params,θa, traj, rms_all, des_traj_real)
run_episode_continous!(params,θa2, traj2, rms_all, des_traj_real)





"""
4. Plot the results of the received trajectory
    => we need to rotate the koordinate system at every step
"""

using HDF5

c = h5open("Results/10_01_real_segway/data.h5", "r") do file
    read(file, "xu")
end

t_start = 10

x_real = Float64.(c)[1:7,:]
u1     = Float64.(c)[8:9,:]
t_real = Float64.(c)[10,:]
len_real = length(t_real)

des_traj_real = zeros(length(x_real[1,:]),2)
iter_beginn = t_start * 100
iter_end    = iter_beginn + length(des_traj[:,1])
l_end = len_real - iter_end +1
des_traj_real[iter_beginn:iter_end-1,:] = des_traj
des_traj_real[iter_end:end,:] = des_traj[end,:]' .* ones(l_end,2)





f_size = 12

y1 = 0.0
y2 = 0
yd = 4



fig = figure("all_res")
clf()
PyPlot.axes([0.12, 0.45, 0.33, 0.5])
title("PPO-Result", fontsize = f_size)
grid(alpha = 0.25)
plot(t_real,des_traj_real[:,1] .+ y1, label="x_des", color="black", linestyle = "--")
plot(t_real,traj.s_t[:,1] .+ y1)
plot(t_real,des_traj_real[:,2] .+ y1, label="y_des", color="black", linestyle = "-.")
plot(t_real,traj.s_t[:,2] .+ y1)
ylim([-0.5, 5])
#xlabel("time (s)")
ylabel("position (m)", fontsize = f_size)

PyPlot.axes([0.12, 0.1, 0.33, 0.25])
grid(alpha = 0.25)
plot(0,0)
plot(0,0)
plot(t_real, traj.a_t[:,1] .+ y2)#, color = "red")
plot(t_real, traj.a_t[:,2] .+ y2)#, color = "blue")
#plot(t_real, traj.a_t[:,3] .+ y2)#, color = "blue")
ylim([-yd+y2, yd+y2])
xlabel("time (s)", fontsize = f_size)
ylabel("u (V)", fontsize = f_size)


#------------------------------------------
PyPlot.axes([0.12+0.43, 0.45, 0.33, 0.5])
title("PPO-soft Result", fontsize = f_size)
grid(alpha = 0.25)
plot(t_real,des_traj_real[:,1] .+ y1, label="x_des", color="black", linestyle = "--")
plot(t_real,traj2.s_t[:,1] .+ y1)
plot(t_real,des_traj_real[:,2] .+ y1, label="y_des", color="black", linestyle = "-.")
plot(t_real,traj2.s_t[:,2] .+ y1)
ylim([-0.5,5])
#xlabel("time (s)")
#ylabel("position (m)")

PyPlot.axes([0.12+0.43, 0.1, 0.33, 0.25])
grid(alpha = 0.25)
plot(0,0)
plot(0,0)
plot(t_real, traj2.a_t[:,1] .+ y2)#, color = "red")
plot(t_real, traj2.a_t[:,2] .+ y2)#, color = "blue")
#plot(t_real, traj2.a_t[:,3] .+ y2)#, color = "blue")
ylim([-yd+y2, yd+y2])
xlabel("time (s)", fontsize = f_size)
#ylabel("u (Nm)")


figure("legend_seg3")
clf()
axis("off")
plot(0,0, label="x_des", color="black", linestyle = "--")
plot(0,0, label="y_des", color="black", linestyle = "-.")
plot(0,0, label = "x_true")
plot(0,0, label = "y_true")
plot(0,0, label = "u_1")
plot(0,0, label = "u_2")
#plot(0,0, label = "u_3")
legend(ncol = 3)


figure("draw")
clf()
plot(des_traj_real[:,1], des_traj_real[:,2])
plot(traj2.s_t[:,1], traj2.s_t[:,2])
plot(traj.s_t[:,1], traj.s_t[:,2])



X  = traj.s_t'
X2 = traj2.s_t'
u  = traj.a_t'
u2 = traj.a_t'
t  = vec(0:0.01:((length(u[1,:])-1) * 0.01))
length(t)

#using Model_Vis
filenameit = "results_ballvsball.jld2"
#visualise_ballbott_compare(X,X2,u,u2,t, filename, des_traj_real; options= Dict())



#For Visualisation in Blender
using NPZ

randomis = 0.001

X_soll = zeros(12,9085)
X_soll[1:7,:]  = copy(X)
X_soll[1:2,:]  = des_traj_real'
X_soll[1,:]   += f_filt(randomis*randn(length(X_soll[1,:])))
X_soll[2,:]   += f_filt(randomis*randn(length(X_soll[1,:])))

X[1,:] += f_filt(randomis*randn(length(X_soll[1,:])))
X[2,:] += f_filt(randomis*randn(length(X_soll[1,:])))

plot(X_soll[:,1])


npzwrite("X_soll4.npy", X_soll)
npzwrite("simulation.npy", X)
npzwrite("experiment.npy", x_real)

npzread("X_soll2.npy")


PPOs = npzread("PPO-soft.npy")
PPO  = npzread("PPO.npy")

PPOs[1,:] += f_filt(randomis*randn(length(PPOs[1,:])))
PPOs[2,:] += f_filt(randomis*randn(length(PPOs[1,:])))

PPO[1,:] += f_filt(randomis*randn(length(PPO[1,:])))
PPO[2,:] += f_filt(randomis*randn(length(PPO[1,:])))

npzwrite("PPO-soft2.npy", PPO)
npzwrite("PPO2.npy", PPOs)
