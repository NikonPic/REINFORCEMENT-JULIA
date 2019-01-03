using JLD
using PyPlot
using FileIO
using Statistics

a = load("Results/Videos/09_24_segway/Actor_Critic_2018-09-24iteartion_930.jld")
a = load("Results/Videos/10_01_segway/Actor_Critic_2018-10-01iteartion_1260.jld")
res = a["res"]
res = res[1:1259] #./ 500

function f_filt(x)
    l = length(x)
    tau = .1
    y = zeros(l)
    y[1] = x[1]

    for i = 2:l
        y[i] = (1-tau)*y[i-1] + tau *x[i]
    end

    return y
end

clf()
figure(2)
grid(alpha = 0.25)
plot(f_filt(res))
xlabel("Iterations", fontsize = 16)
ylabel("Cost", fontsize = 16)
