#Contains all the evaluation stuff
using JLD2
using PyPlot
using FileIO
using Statistics

a = load("Results/11_06_ballbot/1_all.jld2")["result"]
b = load("Results/11_06_ballbot/2_all.jld2")["result"]
c = load("Results/11_06_ballbot/3_all.jld2")["result"]
d = load("Results/11_06_ballbot/4_all.jld2")["result"]
#e = load("Results/Videos/11_06_ballbot/5_all.jld2")["result"]
#d = load("Results/Videos/10_31_swingup/1_all.jld2")["result"]
#e = load("Results/Videos/10_31_swingup/1_all.jld2")["result"]

function f_filt(x)
    l = length(x)
    tau = 0.1
    y = zeros(l)
    y[1] = x[1]

    for i = 2:l
        y[i] = (1-tau)*y[i-1] + tau *x[i]
    end

    return y
end


a_all = 4
iters = 500

RE  = zeros(a_all,iters)
A2C = zeros(a_all,iters)
TNPG= zeros(a_all,iters)
TRPO= zeros(a_all,iters)
PPO = zeros(a_all,iters)
TD3 = zeros(a_all,iters)

RE[1,:]   = a[1,:]
A2C[1,:]  = a[2,:]
TNPG[1,:] = a[3,:]
TRPO[1,:] = a[4,:]
PPO[1,:]  = a[5,:]
TD3[1,:]  = a[6,:]

RE[2,:]   = b[1,:]
A2C[2,:]  = b[2,:]
TNPG[2,:] = b[3,:]
TRPO[2,:] = b[4,:]
PPO[2,:]  = b[5,:]
TD3[2,:]  = b[6,:]

RE[3,:]   = c[1,:]
A2C[3,:]  = c[2,:]
TNPG[3,:] = c[3,:]
TRPO[3,:] = c[4,:]
PPO[3,:]  = c[5,:]
TD3[3,:]  = c[6,:]

RE[4,:]   = d[1,:]
A2C[4,:]  = d[2,:]
TNPG[4,:] = d[3,:]
TRPO[4,:] = d[4,:]
PPO[4,:]  = d[5,:]
TD3[4,:]  = d[6,:]

#=
RE[5,:]   = e[1,:]
A2C[5,:]  = e[2,:]
TNPG[5,:] = e[3,:]
TRPO[5,:] = e[4,:]
PPO[5,:]  = e[5,:]
TD3[5,:]  = e[6,:]
=#

scale  = 1
vis    = 0.5
offset = 0

re_mean = f_filt(vec(mean(RE,dims = 1)) .+ offset)
re_std  = f_filt(scale*vec(std(RE,dims=1)))

a2c_mean = f_filt(vec(mean(A2C,dims = 1)) .+ offset)
a2c_std  = f_filt(scale*vec(std(A2C,dims=1)))

tnpg_mean = f_filt(vec(mean(TNPG,dims = 1)) .+ offset)
tnpg_std  = f_filt(scale*vec(std(TNPG,dims=1)))

trpo_mean = f_filt(vec(mean(TRPO,dims = 1)) .+ offset)
trpo_std  = f_filt(scale*vec(std(TRPO,dims=1)))

ppo_mean = f_filt(vec(mean(PPO,dims = 1)) .+ offset)
ppo_std  = f_filt(scale*vec(std(PPO,dims=1)))

td3_mean = f_filt(vec(mean(TD3,dims = 1)) .+ offset)
td3_std  = f_filt(scale*vec(std(TD3,dims=1)))



figure("ball")
clf()
grid(alpha=0.25)
fill_between(1:iters,re_mean .+ re_std, re_mean .- re_std, alpha=vis)
plot(re_mean, label="REINFORCE")
#fill_between(1:iters,a2c_mean .+ a2c_std, a2c_mean .- a2c_std, alpha=vis)
#plot(a2c_mean, label="A2C")
plot(2,2)
fill_between(1:iters,2,2,2)
#fill_between(1:iters,tnpg_mean .+ tnpg_std, tnpg_mean .- tnpg_std, alpha=vis)
#plot(tnpg_mean, label="TNPG")
plot(2,2)
fill_between(1:iters,2,2,2)
fill_between(1:iters,trpo_mean .+ trpo_std, trpo_mean .- trpo_std, alpha=vis)
plot(trpo_mean, label="TRPO")
fill_between(1:iters,ppo_mean .+ ppo_std, ppo_mean .- ppo_std, alpha=vis)
plot(ppo_mean, label="PPO")
fill_between(1:iters,td3_mean .+ td3_std, td3_mean .- td3_std, alpha=vis)
plot(td3_mean, label="PPO-soft")
xlabel("Iterations", fontsize = 12)
ylabel("Cost", fontsize = 12)
#legend(fontsize = 14)
#ylim([0,15])

RE_min = minimum(RE, dims=2)
RE_min_mean = mean(RE_min)
RE_min_std = std(RE_min)

A2C_min = minimum(A2C, dims=2)
A2C_min_mean = mean(A2C_min)
A2C_min_std = std(A2C_min)

TNPG_min = minimum(TNPG, dims=2)
TNPG_min_mean = mean(TNPG_min)
TNPG_min_std = std(TNPG_min)

TRPO_min = minimum(TRPO, dims=2)
TRPO_min_mean = mean(TRPO_min)
TRPO_min_std = std(TRPO_min)

PPO_min = minimum(PPO, dims=2)
PPO_min_mean = mean(PPO_min)
PPO_min_std = std(PPO_min)

TD3_min = minimum(TD3, dims=2)
TD3_min_mean = mean(TD3_min)
TD3_min_std = std(TD3_min)
