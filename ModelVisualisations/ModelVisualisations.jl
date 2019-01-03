module ModelVisualisations

    using Plots#, HeavyChain

    include("src/invpendulum.jl")
    include("src/ballbot.jl")
    #include("src/heavychain.jl")
    include("src/segwayrobot.jl")

    """
    *function visualise(modelname::Symbol, X::Array{Float64, 2}, t::AbstractVector, filename::String,; options::Dict{Symbol})*\\
    Visualise simulationdata and save .mp4 file.
    Valid modelnames are :InvPendulum, :BallBot, :HeavyChain, :SegwayRobot .
    Valid options are:\\
    - fps: the fps of the video, defaults to 25
    - U: provide input data, which will be plotted below the main window #TODO
    - title: provide a title for the window, simulation time will be added
    - size: resolution of the whole plot in pixels
    """


    function visualise(modelname::Symbol, X::Array{Float64, 2}, t::AbstractVector, filename::String, des_traj::Array{Float64,2}; options::Dict{Symbol}=Dict{Symbol, Any}())
        @assert(issorted(t))
        @assert(length(t)==size(X, 2))

        if modelname == :InvPendulum
            @assert(size(X, 1)==4)
            visualise_invpendulum(X, t, filename; options = options)

        elseif modelname == :BallBot
            @assert(size(X, 1)==12)
            visualise_ballbot(X, t, filename, des_traj; options = options)

        elseif modelname == :HeavyChain
            @assert(size(X, 1)==21)
            visualise_heavychain(X, t, filename; options = options)

        elseif modelname == :SegwayRobot
            @assert(size(X, 1)==7)
            visualise_segwayrobot(X, t, filename, des_traj; options = options)

        end
        return nothing
    end

    export visualise, plot_all_segways, plot_all_ballbots,  visualise_segwayrobot_compare, visualise_ballbot_compare

end # module
