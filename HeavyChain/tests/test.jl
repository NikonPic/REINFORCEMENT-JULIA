using HeavyChain

chain1 = create_chain()
chain2 = create_chain()

dt = 1/1000
t = 0:dt:20
u = sin.(t)

all_xL_1 = Array{Float64}(length(t))
all_xL_2 = Array{Float64}(length(t))

simulate!(chain1, 0.0, dt)
@time for i = 1:length(t)
    simulate!(chain1, u[i], dt)
    all_xL_1[i] = getendpoint(chain1)
end

simulate2!(chain2, 0.0, dt)
@time for i = 1:length(t)
    simulate2!(chain2, u[i], dt)
    all_xL_2[i] = getendpoint(chain2)
end

using Plots
plot(t, all_xL_1, lab="C")
plot!(t, all_xL_2, lab="julia")
