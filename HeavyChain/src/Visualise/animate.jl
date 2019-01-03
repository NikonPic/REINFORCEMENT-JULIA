using Plots
pyplot()

push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../../../../../myJLmodules/LinAlg_C")
using HeavyChain

#some parameters
xmin = -0.2
xmax = 3.8
dt = 1/250
all_t = 0:dt:30
all_u = [NaN for i = 1:length(all_t)]
massmat = zeros(21, 21)
fvec = zeros(21)

#create chain
chain = create_chain()

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
    fig = plot(layout = l, size = (1280, 720))
    heavychain_plot!(fig[1], chain, xmin, xmax, t)
    plot!(fig[2], all_t, all_u, lab = "u")
    frame(anim)
  end

  #simulate one step
  #if chain.q[1]>0.2
  #  u_in = -(chain.q[1]+0.7)^4
  #elseif chain.q[1]<-0.2
  #  u_in = (chain.q[1]-0.7)^4
  #end
  if t<0.1
    u_in = 10.0
  else
    u_in = 0.0
  end

  @time simulate!(chain, u_in, dt, massmat, fvec)
  all_u[iter] = u_in
  iter+=1
end
gif(anim, "animation.mp4", fps = round(Int, 1/0.04))
