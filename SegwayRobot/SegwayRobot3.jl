module SegwayRobot3

    include("src3/segway.jl")
    #export Segway_mutable, Segway_unmutable

    include("src3/dynamics_type.jl")
    include("src3/dynamics_vec.jl")
    #export segway_rk4, segway_rk4!, dxdt_segway

    const params_segway = [-1.23, -1.20, -3.70, -7.51, -14.7, -10.2, -11.9, -3.43]
    const s_init  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    const u_range = [1.0, 1.0]

    include("src3/dynmodel.jl")

    cost_func = segway_cost # set standard Environment cost-function

    function rk4_func(xt::AbstractVector, ut::AbstractVector, dt::Number)

        #run the simulation
        xt1 = dynmodel(xt, ut, dt, params_segway)

        return xt1
    end

    export rk4_func, cost_func, s_init, u_range
end # module SegwayRobot3
