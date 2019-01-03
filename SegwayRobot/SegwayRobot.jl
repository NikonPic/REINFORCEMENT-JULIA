module SegwayRobot

    include("src/dynamics.jl")
    #export segway_rk4, segway_rk4!

    rk4_func  = segway_rk4  # set standard Environment function
    cost_func = segway_cost # set standard Environment cost-function
    #s_init = [0.0, 0.0, 0.0, 0.10] # set standard Environment initial position

    #State vector is [x0, y0, phi, alpha, dalpha, v, dphi, ir, il]
    const s_init  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    const u_range = [11.1, 11.1]
    export rk4_func, cost_func, s_init, u_range
end # module SegwayRoboter
