#Contains all the evaluation stuff
using JLD2
using PyPlot
using FileIO
using Statistics

a= load("Results/10_26_evaluate_pend/1_all.jld2")["RESULT"]
b= load("Results/10_26_evaluate_pend/2_all.jld2")["RESULT"]
c= load("Results/10_26_evaluate_pend/3-4_all.jld2")["RESULT"]

a_all = 4
iters = 500

RE  = zeros(a_all,iters)
A2C = zeros(a_all,iters)
TNPG= zeros(a_all,iters)
TRPO= zeros(a_all,iters)
PPO = zeros(a_all,iters)
TD3 = zeros(a_all,iters)

RE[1,:]   = a[1:6:end]
A2C[1,:]  = a[2:6:end]
TNPG[1,:] = a[3:6:end]
TRPO[1,:] = a[4:6:end]
PPO[1,:]  = a[5:6:end]
TD3[1,:]  = a[6:6:end]

RE[2,:]   = b[1:6:end]
A2C[2,:]  = b[2:6:end]
TNPG[2,:] = b[3:6:end]
TRPO[2,:] = b[4:6:end]
PPO[2,:]  = b[5:6:end]
TD3[2,:]  = b[6:6:end]

RE[3,:]   = c[1:6:3000]
A2C[3,:]  = c[2:6:3000]
TNPG[3,:] = c[3:6:3000]
TRPO[3,:] = c[4:6:3000]
PPO[3,:]  = c[5:6:3000]
TD3[3,:]  = c[6:6:3000]

RE[4,:]   = c[3001:6:end]
A2C[4,:]  = c[3002:6:end]
TNPG[4,:] = c[3003:6:end]
TRPO[4,:] = c[3004:6:end]
PPO[4,:]  = c[3005:6:end]
TD3[4,:]  = c[3006:6:end]

scale = 1
vis   = 0.5
offset = 0

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



figure("pend_v0")
clf()
grid(alpha=0.25)
fill_between(1:iters,re_mean .+ re_std, re_mean .- re_std, alpha=vis)
plot(re_mean, label="REINFORCE")
fill_between(1:iters,a2c_mean .+ a2c_std, a2c_mean .- a2c_std, alpha=vis)
plot(a2c_mean, label="A2C")
fill_between(1:iters,tnpg_mean .+ tnpg_std, tnpg_mean .- tnpg_std, alpha=vis)
plot(tnpg_mean, label="TNPG")
fill_between(1:iters,trpo_mean .+ trpo_std, trpo_mean .- trpo_std, alpha=vis)
plot(trpo_mean, label="TRPO")
fill_between(1:iters,ppo_mean .+ ppo_std, ppo_mean .- ppo_std, alpha=vis)
plot(ppo_mean, label="PPO")
fill_between(1:iters,td3_mean .+ td3_std, td3_mean .- td3_std, alpha=vis)
plot(td3_mean, label="TD3")
xlabel("Iterations", fontsize = 16)
ylabel("Cost", fontsize = 16)
#legend()
