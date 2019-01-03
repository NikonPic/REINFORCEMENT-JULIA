module SegwayRobot2
    using JLD

    include("src2/segway.jl")
    export Segway_mutable, Segway_unmutable

    include("src2/dynamics_type.jl")
    include("src2/dynamics_vec.jl")
    export segway_rk4, segway_rk4!, dxdt_segway

    const nn_params_segway = load("SegwayRobot/src2/params.jld")["nn_params"]
    const s_init  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    const u_range = [1.0, 1.0]

    include("src2/dynamics.jl")

    #rk4_func  = segway_rk4  # set standard Environment function
    cost_func = segway_cost # set standard Environment cost-function

    function rk4_func(xt::AbstractVector, ut::AbstractVector, dt::Number)
        xt1 = copy(xt)
        dynmodel!(xt1, ut, dt, nn_params_segway)
        return xt1
    end

    export rk4_func, cost_func, s_init, u_range
end # module SegwayRobot2
