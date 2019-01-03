module SegwayRobot4

    include("src4/segway.jl")
    export Segway_mutable, Segway_unmutable

    include("src4/dynamics_type.jl")
    include("src4/dynamics_vec.jl")
    #export segway_rk4, segway_rk4!, dxdt_segway
    #x = load("SegwayRobot4/Julia/src/params.jld2")

    const s_init  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    const u_range = [1.0, 1.0]
    const params_segway = [-1.14, -5.09, -4.97, -4.59, -6.68]

    include("src4/dynmodel.jl")

    cost_func = segway_cost # set standard Environment cost-function

    function rk4_func(xt::AbstractVector, ut::AbstractVector, dt::Number)
        xt1 = copy(xt)
        dynmodel!(xt1, ut, params_segway)

        return xt1
    end

    export rk4_func, cost_func, s_init, u_range
end # module SegwayRobot47
