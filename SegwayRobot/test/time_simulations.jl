using SegwayRobot

dt = 1/25
t = 0:dt:20

# Simulate
@time for j = 1:200
    xt = randn(7)*0.01;
    for i = 1:length(t)
        segway_rk4!(xt, (2*xt[1]+50*xt[4]+30*xt[5])*ones(2), dt)
    end
end
