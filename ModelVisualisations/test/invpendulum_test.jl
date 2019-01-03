using ModelVisualisations, InvPendulum, RLExploration

dt = 1/25
t = 0:dt:5
X = zeros(4, length(t))
U = [NaN for i = 1:length(t)]
temp = GaussianProcess()

X[:, 1] .= [0.0, 0.0, 0.0, -pi]
for i = 1:length(t)-1
    U[i] = sample(temp)
    X[:, i+1] .= invpendulum_rk4(X[:, i], U[i], dt)
end

options = Dict(:fps => 25, :size => (1920, 1080), :U => U)
visualise(:InvPendulum, X, t, "invpendulum.mp4"; options = options)
