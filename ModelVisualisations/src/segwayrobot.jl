using ProgressMeter
using Plots

function visualise_segwayrobot(X, t, filename::String, des_traj; options::Dict = Dict())
    # Find plotting limits
    xmin = minimum(X[1, :])-0.1
    xmax = maximum(X[1, :])+0.1
    xlim = (xmin, xmax)
    x_mean = sum(xlim)/2
    x_range = abs.(x_mean-xmin)*2

    ymin = minimum(X[2, :])-0.1
    ymax = maximum(X[2, :])+0.1
    ylim = (ymin, ymax)
    y_mean = sum(ylim) / 2
    y_range = abs.(y_mean-ymin)*2

    #Idea scale equally by having same range:
    range_max = max(x_range, y_range)
    dx = 0.5*(range_max / x_range)*x_range
    dy = 0.5*(range_max / y_range)*y_range

    xlim = (x_mean - dx, x_mean + dx)
    ylim = (y_mean - dy, y_mean + dy)



    fps::Int = get(options, :fps, 25)

    # Create animation
    plotsize::Tuple{Int, Int} = get(options, :size, (1280, 720))
    anim = Animation()

    # Create timestamps of the visualisation and interpolate states
    t_plot = t[1]:(1/fps):t[end]
    X_plot = Array{Float64}(size(X, 1), length(t_plot))


    if haskey(options, :U)
        # Animate with U
        U = get(options, :U, zeros(2, length(t_plot)))
        @assert(size(U, 2)==length(t))
        umin = minimum(U)-0.2
        umax = maximum(U)+0.2
        U_plot = Array{Float64}(2, length(t_plot))
        for (i, t2) = enumerate(t_plot)
            # find closest t's belonging to t_plot[i]
            j = indmin(abs.(t2-t))
            if t2 == t[j]
                @. X_plot[:, i] = X[:, j]
                U_plot[:, i] .= U[:, j]
            elseif t2>t[j]
                @. X_plot[:, i] = (X[:, j+1]-X[:, j])*(t2-t[j])/(t[j+1]-t[j])
                U_plot[:, i] .= (U[:, j+1]-U[:, j])*(t2-t[j])/(t[j+1]-t[j])
            else
                @. X_plot[:, i] = (X[:, j]-X[:, j-1])*(t2-t[j-1])/(t[j]-t[j-1])
                U_plot[:, i] .= (U[:, j]-U[:, j-1])*(t2-t[j-1])/(t[j]-t[j-1])
            end
        end

        #=
        for (i, t2) = enumerate(t_plot)
            #fig = plot(;size = plotsize, layout = grid(2, 1, heights=[0.8,0.2]))
            fig = plot(0,0)
            plot_segway_3d!(fig, X_plot[:, i], xlim, ylim, des_traj, i, X)
            #plot!(fig[2], t_plot, vcat(U_plot[1, 1:i], [NaN for j = i+1:length(t_plot)]), xlim = (t_plot[1], t_plot[end]), ylim = (umin, umax), lab = "u1")
			#plot!(fig[2], t_plot, vcat(U_plot[2, 1:i], [NaN for j = i+1:length(t_plot)]), lab = "u2")
            frame(anim)
        end
        =#
        prog = Progress(length(t))
        n_step = 4
        for i = 1:n_step:length(t)
            fig = plot(;size = plotsize, layout = grid(2, 1, heights=[0.8,0.2]))
            plot_segway_3d!(fig[1], X[:,i], xlim, ylim, des_traj, i, X)
            plot!(fig[2], t[1:i], U[1,1:i], lab = "u1", xlim = (0, t[end]), ylim = (-1, 1))
            plot!(fig[2], t[1:i], U[2,1:i], lab = "u2")
            frame(anim)

            ProgressMeter.update!(prog,i)
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
            fig = plot_segway_3d(X_plot[:, i], xlim, ylim, plotsize)
            frame(anim)
        end
    end
    run(`ffmpeg -v 0 -framerate $(fps) -loop 0 -i $(anim.dir)/%06d.png -y $(filename)`)
    return nothing
end


function plot_segway_3d(xt, xlim, ylim, size)
    fig = plot(size=size)
    plot_segway_3d!(fig, xt, xlim, ylim)
    return fig
end

function plot_segway_3d!(fig, xt, xlim, ylim, des_traj, i, X)
    #Trajectory to do:
    plot!(fig,des_traj[1:i,1],des_traj[1:i,2],des_traj[1:i,1] .*0, lab ="desired position")

    #Trajectory done:
    plot!(fig,X[1,1:i],X[2,1:i],X[1,1:i].*0, lab="true position")

    # Constants (copied from dynamics)
    R = 0.033 # m Radius der Reifen?
    b = 0.098/2 # m halber Abstand zwischen den Reifen?
    cz = 0.04867 # m #TODO R?3R #Schwerpunkt Höhe

    # Rotation matrices
    Rmat_y = [cos(xt[4]) 0 sin(xt[4]);
              0 1 0;
              -sin(xt[4]) 0 cos(xt[4])]

    Rmat_z = [cos(xt[3]) -sin(xt[3]) 0;
            sin(xt[3]) cos(xt[3]) 0;
            0 0 1]

    # Plot floor
    #X=repeat(xlim, outer = (1, 2))
    #Y=repeat(ylim', outer = (2, 1))
    #surface!(fig, X, Y, zeros(size(X)), zlim = (-0.01, 2.5*cz), lab = "", colorbar = :green)
    range = xlim[2] - xlim[1]
    zlim = (-0.01, max(3*cz,-0.01+range))
    plot!(fig, xlim = xlim, ylim = ylim, zlim = zlim, ratio = :equal) # TODO zlim problematesch

    # Plot tires
    t = linspace(0, 2*pi, 20)
    xtire = R*cos.(t)
    ztire = R+R*sin.(t)

    centeroffset = [0.0, b, 0.0]
    tirecoords = Array{Float64}(3, length(t))
    for i = 1:size(tirecoords, 2)
        tirecoords[:, i] = Rmat_z*vcat(xtire[i] + centeroffset[1], 0.0 + centeroffset[2], ztire[i] + centeroffset[3])
    end
    plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:black, w = 3, lab = "")
    for i = 1:size(tirecoords, 2)
        tirecoords[:, i] = Rmat_z*vcat(xtire[i] - centeroffset[1], 0.0 - centeroffset[2], ztire[i] - centeroffset[3])
    end
    plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:black, w = 3, lab = "")

    # Plot body as a grid
    facey = [-b, b, b, -b, -b]
    facez = [0.0, 0.0, 0.0+2*cz+R, 0.0+2*cz+R, 0.0]
    bodycoords = Array{Float64}(3, length(facey))
    for i = 1:size(bodycoords, 2)
        bodycoords[:, i] = Rmat_z*Rmat_y*vcat(0.0, facey[i], facez[i])
    end
    plot!(fig, bodycoords[1, :] + xt[1], bodycoords[2, :] + xt[2], bodycoords[3, :] + R,  c=:black, w = 5, lab = "")

end


function plot_all_segways(all_traj,filename::String, params; options::Dict = Dict())
    # Find plotting limits
    xmin = -1
    xmax = +1
    xlim = (xmin, xmax)
    x_mean = sum(xlim)/2
    x_range = abs.(x_mean-xmin)*2

    ymin = -1
    ymax = +1
    ylim = (ymin, ymax)
    y_mean = sum(ylim) / 2
    y_range = abs.(y_mean-ymin)*2

    #Idea scale equally by having same range:
    range_max = max(x_range, y_range)
    dx = 0.5*(range_max / x_range)*x_range
    dy = 0.5*(range_max / y_range)*y_range

    xlim = (x_mean - dx, x_mean + dx)
    ylim = (y_mean - dy, y_mean + dy)

    #Preselect good trajectory
    max_len = 0
    for i_traj = 1: params.n_traj
        if  all_traj.traj[i_traj].len > max_len
            max_len = all_traj.traj[i_traj].len
            i_best =  i_traj
        end
    end

    #Take data for video
    t_plot = 0:params.dt:params.t_horizon;
    t_plot = t_plot[1:max_len]



    fps::Int = get(options, :fps, 25)

    # Create animation
    plotsize::Tuple{Int, Int} = get(options, :size, (1280, 720))
    anim = Animation()

    n_step = 10
    n_time = 4

    des_traj = zeros(2,2)
    i = 1
    X = zeros(2,2)

    #Go over all timesteps
    for i_t = 1:n_time:max_len
        fig = plot(size=plotsize)

        #Take all selected trajectories
        for i_traj = 1:n_step:params.n_traj

            #Check if trajectory is still active
            if all_traj.traj[i_traj].len >= i_t
                xt = all_traj.traj[i_traj].s_t[i_t,:]
                plot_segway_3d!(fig, xt, xlim, ylim, des_traj,i,X)
            end
        end
        frame(anim)
    end

    run(`ffmpeg -v 0 -framerate $(fps) -loop 0 -i $(anim.dir)/%06d.png -y $(filename)`)
    return nothing
end




function plot_segway_3d_compare!(fig, xt, xt2, xlim, ylim, des_traj, i, X1, X2)
    #Trajectory to do:
    plot!(fig,des_traj[1:i,1],des_traj[1:i,2],des_traj[1:i,1] .*0, lab ="desired position", c=:black)

    #Trajectory done:
    plot!(fig,X1[1,1:i],X1[2,1:i],X1[1,1:i].*0, lab="true position simulation", c=:blue)

    #Trajectory done:
    plot!(fig,X2[1,1:i],X2[2,1:i],X2[1,1:i].*0, lab="true position experiment", c=:green)

    p_close = [2.0, 0.0]

    d_sim  = (p_close[1] - xt[1] )^2 + (p_close[2] - xt[2] )^2
    d_real = (p_close[1] - xt2[1])^2 + (p_close[2] - xt2[2])^2

    # Constants (copied from dynamics)
    R = 0.033 # m Radius der Reifen?
    b = 0.098/2 # m halber Abstand zwischen den Reifen?
    cz = 0.04867 # m #TODO R?3R #Schwerpunkt Höhe

    if d_sim > d_real
        # Rotation matrices
        Rmat_y = [cos(xt[4]) 0 sin(xt[4]);
                  0 1 0;
                  -sin(xt[4]) 0 cos(xt[4])]

        Rmat_z = [cos(xt[3]) -sin(xt[3]) 0;
                sin(xt[3]) cos(xt[3]) 0;
                0 0 1]

        # Plot floor
        #X=repeat(xlim, outer = (1, 2))
        #Y=repeat(ylim', outer = (2, 1))
        #surface!(fig, X, Y, zeros(size(X)), zlim = (-0.01, 2.5*cz), lab = "", colorbar = :green)
        range = xlim[2] - xlim[1]
        zlim = (-0.01, max(3*cz,-0.01+range))
        plot!(fig, xlim = xlim, ylim = ylim, zlim = zlim, ratio = :equal) # TODO zlim problematesch

        # Plot tires
        t = linspace(0, 2*pi, 20)
        xtire = R*cos.(t)
        ztire = R+R*sin.(t)

        centeroffset = [0.0, b, 0.0]
        tirecoords = Array{Float64}(3, length(t))
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] + centeroffset[1], 0.0 + centeroffset[2], ztire[i] + centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:blue, w = 3, lab = "")
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] - centeroffset[1], 0.0 - centeroffset[2], ztire[i] - centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:blue, w = 3, lab = "")

        # Plot body as a grid
        facey = [-b, b, b, -b, -b]
        facez = [0.0, 0.0, 0.0+2*cz+R, 0.0+2*cz+R, 0.0]
        bodycoords = Array{Float64}(3, length(facey))
        for i = 1:size(bodycoords, 2)
            bodycoords[:, i] = Rmat_z*Rmat_y*vcat(0.0, facey[i], facez[i])
        end
        plot!(fig, bodycoords[1, :] + xt[1], bodycoords[2, :] + xt[2], bodycoords[3, :] + R,  c=:blue, w = 5, lab = "")

        """
        Second Segway
        """

        # Rotation matrices
        Rmat_y = [cos(xt2[4]) 0 sin(xt2[4]);
              0 1 0;
              -sin(xt2[4]) 0 cos(xt2[4])]

        Rmat_z = [cos(xt2[3]) -sin(xt2[3]) 0;
            sin(xt2[3]) cos(xt2[3]) 0;
            0 0 1]

        # Plot floor
        #X=repeat(xlim, outer = (1, 2))
        #Y=repeat(ylim', outer = (2, 1))
        #surface!(fig, X, Y, zeros(size(X)), zlim = (-0.01, 2.5*cz), lab = "", colorbar = :green)
        range = xlim[2] - xlim[1]
        zlim = (-0.01, max(3*cz,-0.01+range))
        plot!(fig, xlim = xlim, ylim = ylim, zlim = zlim, ratio = :equal) # TODO zlim problematesch

        # Plot tires
        t = linspace(0, 2*pi, 20)
        xtire = R*cos.(t)
        ztire = R+R*sin.(t)

        centeroffset = [0.0, b, 0.0]
        tirecoords = Array{Float64}(3, length(t))
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] + centeroffset[1], 0.0 + centeroffset[2], ztire[i] + centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt2[1], tirecoords[2, :]+xt2[2], tirecoords[3, :], c=:green, w = 3, lab = "")
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] - centeroffset[1], 0.0 - centeroffset[2], ztire[i] - centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt2[1], tirecoords[2, :]+xt2[2], tirecoords[3, :], c=:green, w = 3, lab = "")

        # Plot body as a grid
        facey = [-b, b, b, -b, -b]
        facez = [0.0, 0.0, 0.0+2*cz+R, 0.0+2*cz+R, 0.0]
        bodycoords = Array{Float64}(3, length(facey))
        for i = 1:size(bodycoords, 2)
            bodycoords[:, i] = Rmat_z*Rmat_y*vcat(0.0, facey[i], facez[i])
        end
        plot!(fig, bodycoords[1, :] + xt2[1], bodycoords[2, :] + xt2[2], bodycoords[3, :] + R,  c=:green, w = 5, lab = "")

    else

        # Rotation matrices
        Rmat_y = [cos(xt2[4]) 0 sin(xt2[4]);
                  0 1 0;
                  -sin(xt2[4]) 0 cos(xt2[4])]

        Rmat_z = [cos(xt2[3]) -sin(xt2[3]) 0;
                sin(xt2[3]) cos(xt2[3]) 0;
                0 0 1]

        # Plot floor
        #X=repeat(xlim, outer = (1, 2))
        #Y=repeat(ylim', outer = (2, 1))
        #surface!(fig, X, Y, zeros(size(X)), zlim = (-0.01, 2.5*cz), lab = "", colorbar = :green)
        range = xlim[2] - xlim[1]
        zlim = (-0.01, max(3*cz,-0.01+range))
        plot!(fig, xlim = xlim, ylim = ylim, zlim = zlim, ratio = :equal) # TODO zlim problematesch

        # Plot tires
        t = linspace(0, 2*pi, 20)
        xtire = R*cos.(t)
        ztire = R+R*sin.(t)

        centeroffset = [0.0, b, 0.0]
        tirecoords = Array{Float64}(3, length(t))
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] + centeroffset[1], 0.0 + centeroffset[2], ztire[i] + centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt2[1], tirecoords[2, :]+xt2[2], tirecoords[3, :], c=:green, w = 3, lab = "")
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] - centeroffset[1], 0.0 - centeroffset[2], ztire[i] - centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt2[1], tirecoords[2, :]+xt2[2], tirecoords[3, :], c=:green, w = 3, lab = "")

        # Plot body as a grid
        facey = [-b, b, b, -b, -b]
        facez = [0.0, 0.0, 0.0+2*cz+R, 0.0+2*cz+R, 0.0]
        bodycoords = Array{Float64}(3, length(facey))
        for i = 1:size(bodycoords, 2)
            bodycoords[:, i] = Rmat_z*Rmat_y*vcat(0.0, facey[i], facez[i])
        end
        plot!(fig, bodycoords[1, :] + xt2[1], bodycoords[2, :] + xt2[2], bodycoords[3, :] + R,  c=:green, w = 5, lab = "")



        # Rotation matrices
        Rmat_y = [cos(xt[4]) 0 sin(xt[4]);
                  0 1 0;
                  -sin(xt[4]) 0 cos(xt[4])]

        Rmat_z = [cos(xt[3]) -sin(xt[3]) 0;
                sin(xt[3]) cos(xt[3]) 0;
                0 0 1]

        # Plot floor
        #X=repeat(xlim, outer = (1, 2))
        #Y=repeat(ylim', outer = (2, 1))
        #surface!(fig, X, Y, zeros(size(X)), zlim = (-0.01, 2.5*cz), lab = "", colorbar = :green)
        range = xlim[2] - xlim[1]
        zlim = (-0.01, max(3*cz,-0.01+range))
        plot!(fig, xlim = xlim, ylim = ylim, zlim = zlim, ratio = :equal) # TODO zlim problematesch

        # Plot tires
        t = linspace(0, 2*pi, 20)
        xtire = R*cos.(t)
        ztire = R+R*sin.(t)

        centeroffset = [0.0, b, 0.0]
        tirecoords = Array{Float64}(3, length(t))
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] + centeroffset[1], 0.0 + centeroffset[2], ztire[i] + centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:blue, w = 3, lab = "")
        for i = 1:size(tirecoords, 2)
            tirecoords[:, i] = Rmat_z*vcat(xtire[i] - centeroffset[1], 0.0 - centeroffset[2], ztire[i] - centeroffset[3])
        end
        plot!(fig, tirecoords[1, :]+xt[1], tirecoords[2, :]+xt[2], tirecoords[3, :], c=:blue, w = 3, lab = "")

        # Plot body as a grid
        facey = [-b, b, b, -b, -b]
        facez = [0.0, 0.0, 0.0+2*cz+R, 0.0+2*cz+R, 0.0]
        bodycoords = Array{Float64}(3, length(facey))
        for i = 1:size(bodycoords, 2)
            bodycoords[:, i] = Rmat_z*Rmat_y*vcat(0.0, facey[i], facez[i])
        end
        plot!(fig, bodycoords[1, :] + xt[1], bodycoords[2, :] + xt[2], bodycoords[3, :] + R,  c=:blue, w = 5, lab = "")

    end

end







function visualise_segwayrobot_compare(X,X2,u,u2,t, filename::String, des_traj; options::Dict = Dict())
    # Find plotting limits
    xmin = minimum(X[1, :])-0.1
    xmax = maximum(X[1, :])+0.1
    xlim = (xmin, xmax)
    x_mean = sum(xlim)/2
    x_range = abs.(x_mean-xmin)*2

    ymin = minimum(X[2, :])-0.1
    ymax = maximum(X[2, :])+0.1
    ylim = (ymin, ymax)
    y_mean = sum(ylim) / 2
    y_range = abs.(y_mean-ymin)*2

    #Idea scale equally by having same range:
    range_max = max(x_range, y_range)
    dx = 0.5*(range_max / x_range)*x_range
    dy = 0.5*(range_max / y_range)*y_range

    xlim = (x_mean - dx, x_mean + dx)
    ylim = (y_mean - dy, y_mean + dy)



    fps::Int = get(options, :fps, 25)

    # Create animation
    plotsize::Tuple{Int, Int} = get(options, :size, (1280, 720))
    anim = Animation()

    # Create timestamps of the visualisation and interpolate states
    t_plot = t[1]:(1/fps):t[end]
    X_plot = Array{Float64}(size(X, 1), length(t_plot))


    prog = Progress(length(t))
    n_step = 4
    for i = 1:n_step:length(t)

        fig = plot(;size = plotsize, layout = grid(3, 1, heights=[0.8,0.1,0.1]))
        plot_segway_3d_compare!(fig[1], X[:,i], X2[:,i], xlim, ylim, des_traj, i, X, X2)
        plot!(fig[2], t[1:i], u[1,1:i], lab = "u1_sim", xlim = (0, t[end]), ylim = (-1, 1))
        plot!(fig[2], t[1:i], u[2,1:i], lab = "u2_sim")
        plot!(fig[3], t[1:i], u2[1,1:i], lab = "u1_exp", xlim = (0, t[end]), ylim = (-1, 1))
        plot!(fig[3], t[1:i], u2[2,1:i], lab = "u2_exp")
        frame(anim)

        ProgressMeter.update!(prog,i)
    end

    run(`ffmpeg -v 0 -framerate $(fps) -loop 0 -i $(anim.dir)/%06d.png -y $(filename)`)
    return nothing
end
