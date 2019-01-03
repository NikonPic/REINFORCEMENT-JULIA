using Plots
pyplot()

include("../heavy_chain.jl")
include("../heavy_chain_plot.jl")

#some parameters
xmin = -1
xmax = 1
dt = 1/1000
all_t = 0:dt:40
all_u = [NaN for i = 1:length(all_t)]
x_coords = collect(linspace(0, 1.2, 200))

#create chain
chain = Chain(x_coords)

#define input u(t)
#u_in(t) = mod(t, 3)>1.0 ? 1.0:-1.0

#simulate and animate
anim = Animation()
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
  if chain.w[1]>0.2
    u_in = -(chain.w[1]+0.7)^4
  elseif chain.w[1]<-0.2
    u_in = (chain.w[1]-0.7)^4
  end
  simulate!(chain, u_in, dt)
  all_u[iter] = u_in
  iter+=1
end
gif(anim, "animation.mp4", fps = round(Int, 1/0.04))
