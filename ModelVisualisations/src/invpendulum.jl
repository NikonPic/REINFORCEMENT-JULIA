#using InvPendulum

function visualise_invpendulum(X, t, filename::String; options::Dict = Dict())
    # Find plotting limits
    xmin = minimum(X[2, :])-0.67
    xmax = maximum(X[2, :])+0.67
    fps::Int = get(options, :fps, 25)

    # Create animation
    title::String = get(options, :title, "")
    size::Tuple{Int, Int} = get(options, :size, (1280, 720))
    anim = Animation()

    # Create timestamps of the visualisation and interpolate states
    t_plot = t[1]:(1/fps):t[end]
    X_plot = Array{Float64}(4, length(t_plot))


    if haskey(options, :U)
        # Animate with U
        U = get(options, :U, zeros(length(t_plot)))
        @assert(length(U)==length(t))
        umin = minimum(U)-0.2
        umax = maximum(U)+0.2
        U_plot = Array{Float64}(length(t_plot))
        for (i, t2) = enumerate(t_plot)
            # find closest t's belonging to t_plot[i]
            j = indmin(abs.(t2-t))
            if t2 == t[j]
                @. X_plot[:, i] = X[:, j]
                U_plot[i] = U[j]
            elseif t2>t[j]
                @. X_plot[:, i] = (X[:, j+1]-X[:, j])*(t2-t[j])/(t[j+1]-t[j])
                U_plot[i] = (U[j+1]-U[j])*(t2-t[j])/(t[j+1]-t[j])
            else
                @. X_plot[:, i] = (X[:, j]-X[:, j-1])*(t2-t[j-1])/(t[j]-t[j-1])
                U_plot[i] = (U[j]-U[j-1])*(t2-t[j-1])/(t[j]-t[j-1])
            end
        end

        for (i, t2) = enumerate(t_plot)
            fig = plot(;size = size, layout = grid(2, 1, heights=[0.8,0.2]))
            plot_pendulum!(fig[1], X_plot[:, i], t2, xmin, xmax, title)
            plot!(fig[2], t_plot, vcat(U_plot[1:i], [NaN for j = i+1:length(t_plot)]), ylim = (umin, umax), lab = "u")
            frame(anim)
        end
    else
        # Animate without U
        for (i, t2) = enumerate(t_plot)
            # find closest t's belonging to t_plot[i]
            j = indmin(abs.(t2-t))
            if t2 == t[j]
                @. X_plot[:, i] = X[:, j]
            elseif t2>t[j]
                @. X_plot[:, i] = (X[:, j+1]-X[:, j])*(t2-t[j])/(t[j+1]-t[j])
            else
                @. X_plot[:, i] = (X[:, j]-X[:, j-1])*(t2-t[j-1])/(t[j]-t[j-1])
            end
        end

        for (i, t2) = enumerate(t_plot)
            fig = plot_pendulum(X_plot[:, i], t2, xmin, xmax, size, title)
            frame(anim)
        end
    end
    run(`ffmpeg -v 0 -framerate $(fps) -loop 0 -i $(anim.dir)/%06d.png -y $(filename)`)
    return nothing
end


#plotting the pendulum, xmin and xmax indicate how far the pendulum can move
function plot_pendulum!(fig, state::Array{Float64, 1}, time::Float64, xmin::Number, xmax::Number, title::String)
    l = 0.3302*2 #length of the rod
    lw = 0.015 #thickness of the rod
    basewidth = 0.2 #width of the car
    baseheight = 0.15 #height of the car

    #base plot
    Plots.plot!(fig, ratio=:equal, xlim = (xmin, xmax), ylim = (-l*1.1, l*1.1), title = title*" t=$(round(time, 1))", box=:full, grid=false)

    #plot zero line
    Plots.plot!(fig, [xmin, xmax], [0, 0], c=:black, l=:dot, lab = "", grid = false)

    #plot base
    base(x) = Plots.Shape(x+[-basewidth/2, basewidth/2, basewidth/2, -basewidth/2, -basewidth/2], [-baseheight/2, -baseheight/2, baseheight/2, baseheight/2, -baseheight/2])
    Plots.plot!(fig, base(state[2]), c=:orange, lab = "")

    #plot rod
    rot_mat = [cos(state[4]) -sin(state[4]); sin(state[4])  cos(state[4])] #rotation matrix, rotating the rod
    corners = [-lw/2 lw/2 lw/2 -lw/2 -lw/2; -l*0.03 -l*0.03 l l -l*0.03]
    corners = rot_mat*corners

    xrod = corners[1, :]
    yrod = corners[2, :]
    rod(x) = Plots.Shape(x+xrod, yrod)
    Plots.plot!(fig, rod(state[2]), c=:green, lab = "")
end
#plot_pendulum!(fig, state::Ptr{vector}, time::Float64, xmin::Number, xmax::Number) = plot_pendulum!(fig, convert(Array{Float64, 1}, state), time, xmin, xmax)


function plot_pendulum(state::Array{Float64, 1}, time::Float64, xmin::Number, xmax::Number, size::Tuple{Int, Int}, title::String)
    fig = plot(size=size)
    plot_pendulum!(fig, state, time, xmin, xmax, title)
    return fig
end
