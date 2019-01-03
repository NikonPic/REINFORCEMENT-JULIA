# Simulate a falling segway
using SegwayRobot, Plots

xt = zeros(7)
xt[1:2] .= randn(2)
xt[4] = pi/12

dt = 1/25
t = 0:dt:30

anim = Animation()
for i = 1:length(t)
    # Plot robot
    fig = plot_segway_3d(xt, [-1, 19], [-10, 10])
    plot!(fig, title = "t = $(round(t[i],1))", xlab = "x", ylab = "y", size = (1920, 1080))
    frame(anim)

    # Simulate
    ut = (0.3*xt[1]+50*xt[4]+30*xt[5]+5*xt[6])*ones(2)
	#println(ut)
    segway_rk4!(xt, ut, dt)
end

# Create animation
run(`ffmpeg -v 0 -framerate $(round(Int, 1/dt)) -loop 0 -i $(anim.dir)/%06d.png -y video_pcontrol.mp4`)
