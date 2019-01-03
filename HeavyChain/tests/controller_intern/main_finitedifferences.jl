#load pendulum and controllers
include("../FiniteDifferences/heavy_chain.jl")
include("feedforward.jl")
include("feedback.jl")

#define parameters
enable_anim = false
enable_sim = true
#x_coords = collect(linspace(0, 1.178, 200))
x_coords = collect(1.0-(logspace(1, 0, 200)-1)./9)*1.178
chain = Chain(x_coords)
dt = 1/2000
all_t = collect(0:dt:15)
(F_ff, xL_target, x0_target, v0_target, phi0_target, phi0_dot_target) = chain_feedforward(chain, all_t)

#plot signals
using Plots
pyplot()
fig = plot(layout = (3, 1))
plot!(fig[1], all_t, xL_target, lab = "xL")
plot!(fig[1], all_t, x0_target, l = :dot, lab = "x0")
plot!(fig[2], all_t, phi0_target, lab = "phi")
plot!(fig[2], all_t, phi0_dot_target, l = :dot, lab = "dphidt")
plot!(fig[3], all_t, F_ff, lab = "F")
gui()

#simulate
if enable_sim
  println("Starting simulation")
  all_xL = Array{Float64}(length(xL_target))
  all_x0 = Array{Float64}(length(x0_target))
  for i = 1:length(all_t)
    u_in = F_ff[i]
    all_x0[i] = chain.w[1]
    all_xL[i] = chain.w[end]
    simulate!(chain, u_in, dt)
  end
  fig = plot(layout = (2, 1))
  plot!(fig[1], all_t, x0_target, l=:dot, lab = "x0_target")
  plot!(fig[1], all_t, all_x0, lab = "x0", c=:black)

  plot!(fig[2], all_t, xL_target, l=:dot, lab = "xL_target")
  plot!(fig[2], all_t, all_xL, lab = "xL", c=:black)
end


#animate
if enable_anim
  println("Starting animation")
  include("../heavy_chain_plot.jl")
  xmin = -0.2
  xmax = 0.7
  chain = Chain(x_coords)
  anim = Animation()
  all_u = [NaN for i = 1:length(all_t)]
  iter=1
  u_in = 1.0
  for t = all_t
    #take frame
    if mod(t, 0.04)<dt*0.5 || mod(t, 0.04)>(0.04-dt*0.5)
      println("Frame at t = $(t)")
      l = Plots.@layout [a{0.7h}; b]
      fig = plot(layout = l)
      heavy_chain_plot!(fig[1], chain, xmin, xmax, t)
      plot!(fig[2], all_t, all_u, lab = "u")
      frame(anim)
    end

    #simulate one step
    u_in = F_ff[iter]
    simulate!(chain, u_in, dt)
    all_u[iter] = u_in
    iter+=1
  end
  gif(anim, "animation.mp4", fps = round(Int, 1/0.04))
end
