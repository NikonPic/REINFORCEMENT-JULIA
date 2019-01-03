module InvPendulum

    include("src/invpend_julia.jl")
    rk4_func  = inv_pendulum_rk4  # set standard Environment function
    cost_func = inv_pendulum_cost#_swingup # set standard Environment cost-function
    s_init = [0.0, 0.0, 0.0, 0.0] # set standard Environment initial position
    #s_init = [0.0, 0.0, 0.0, pi]
    u_range = [11.1]
    export rk4_func, cost_func, s_init, u_range

end
