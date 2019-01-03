#load pendulum and controllers
using HeavyChain
include("feedforward.jl")
include("feedback2.jl")

#define parameters
enable_anim = false
enable_sim = true
chain = create_chain()
dt = 1/1000
all_t = collect(0:dt:15)
(F_ff, xL_target, x0_target, v0_target, phi0_target, phi0_dot_target) = chain_feedforward(all_t)

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
u_in = 0.0
if enable_sim
  println("Starting simulation")
  chain = create_chain()
  all_xL = Array{Float64}(length(xL_target))
  all_x0 = Array{Float64}(length(x0_target))
  @time for i = 1:length(all_t)
    if mod(i, 5)==1
      target = vcat(x0_target[i], v0_target[i], phi0_target[i], phi0_dot_target[i])
      u_in = chain_feedback(chain, target) #+F_ff[i]
    end
    all_x0[i] = chain.q[1]
    all_xL[i] = getendpoint(chain)
    simulate!(chain, u_in, dt)
  end
  fig = plot(layout = (2, 1))
  plot!(fig[1], all_t, x0_target, l=:dot, lab = "x0_target")
  plot!(fig[1], all_t, all_x0, lab = "x0", c=:black)

  plot!(fig[2], all_t, xL_target, l=:dot, lab = "xL_target")
  plot!(fig[2], all_t, all_xL, lab = "xL", c=:black)
  savefig("plot.pdf")
end


#animate
if enable_anim
  println("Starting animation")
  xmin = -0.2
  xmax = 0.7
  chain = create_chain()
  anim = Animation()
  all_u = [NaN for i = 1:length(all_t)]
  iter=1
  u_in = 0.0
  for t = all_t
    #take frame
    if mod(t, 0.04)<dt*0.5 || mod(t, 0.04)>(0.04-dt*0.5)
      println("Frame at t = $(t)")
      l = Plots.@layout [a{0.7h}; b]
      fig = plot(layout = l, size = (1280, 720))
      heavychain_plot!(fig[1], chain, xmin, xmax, t)
      plot!(fig[2], all_t, all_u, lab = "u")
      frame(anim)
    end

    #simulate one step
    if mod(iter, 5)==1
      target = vcat(x0_target[iter], v0_target[iter], phi0_target[iter], phi0_dot_target[iter])
      u_in = F_ff[iter]#+chain_feedback(chain, target)
    end
    simulate!(chain, u_in, dt, massmat, fvec)
    all_u[iter] = u_in
    iter+=1
  end
  gif(anim, "animation.mp4", fps = round(Int, 1/0.04))
end
