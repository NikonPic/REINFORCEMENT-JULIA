using Plots
pyplot()
using Statistics
using ProgressMeter

function visualise_ballbott_compare(X,X2,u,u2,t, filename::String, des_traj; options::Dict = Dict())
    # Find plotting limits
    xmin = minimum(X[1, :]; dims = 1)[1] .- .2
    xmax = maximum(X[1, :]; dims = 1)[1] .+ .2
    xlim = (xmin, xmax)
    x_mean = sum(xlim)/2
    x_range = abs.(x_mean-xmin)*2

    ymin = minimum(X[2, :])#-2
    ymax = maximum(X[2, :])#+2
    ylim = (ymin, ymax)
    y_mean = sum(ylim) / 2
    y_range = abs.(y_mean-ymin)*2

    #Idea scale equally by having same range:
    range_max = max.(x_range, y_range)
    dx = 0.5*(range_max / x_range)*x_range
    dy = 0.5*(range_max / y_range)*y_range

    xlim = (x_mean - dx, x_mean + dx)
    ylim = (y_mean - dy, y_mean + dy)


    fps::Int = get(options, :fps, 25)

    # Create animation
    plotsize::Tuple{Int, Int} = get(options, :size, (1280, 720))
    #plotsize::Tuple{Int, Int} = get(options, :size, (640, 360))
    anim = Animation()

    # Create timestamps of the visualisation and interpolate states
    t_plot = t[1]:(1/fps):t[end]
    #X_plot = Array{Float64}(size(X, 1), length(t_plot))


    prog = Progress(length(t))
    n_step = 12
    for i = 1:n_step:length(t)

        fig = plot(;size = plotsize, layout = grid(3, 1, heights=[0.8,0.1,0.1]))
        plot_ballbot_3d_compare!(fig[1], X[:,i], X2[:,i], xlim, ylim, des_traj, i, X, X2)
        plot!(fig[2], t[1:i], u[1,1:i], xlim = (0, t[end]), ylim = (-4, 4), lab = "u1_hard" )
        plot!(fig[2], t[1:i], u[2,1:i], lab = "u2_hard")
        plot!(fig[2], t[1:i], u[3,1:i], lab = "u3_hard")
        plot!(fig[3], t[1:i], u2[1,1:i], xlim = (0, t[end]), ylim = (-4, 4), lab = "u1_soft")
        plot!(fig[3], t[1:i], u2[2,1:i], lab = "u2_soft")
        plot!(fig[3], t[1:i], u2[2,1:i], lab = "u3_soft")
        frame(anim)

        ProgressMeter.update!(prog,i)
    end

    run(`ffmpeg -v 0 -framerate $(fps) -loop 0 -i $(anim.dir)/%06d.png -y $(filename)`)
    return nothing
end




function plot_ballbot_3d_compare!(fig, xt, xt2, x_lim, y_lim, des_traj, i, X1, X2)
    #Trajectory to do:
    plot!(fig,des_traj[1:i,1],des_traj[1:i,2],des_traj[1:i,1] .*0, lab ="desired position", c=:black)

    #Trajectory done:
    plot!(fig,X1[1,1:i],X1[2,1:i],X1[1,1:i].*0, lab="position PPO", c=:blue)

    #Trajectory done:
    plot!(fig,X2[1,1:i],X2[2,1:i],X2[1,1:i].*0, lab="position PPO-soft", c=:green)

    p_close = [2.0, 0.0]

    d_sim  = (p_close[1] - xt[1] )^2 + (p_close[2] - xt[2] )^2
    d_real = (p_close[1] - xt2[1])^2 + (p_close[2] - xt2[2])^2

    scale_it = 0.25
    range = x_lim[2] - x_lim[1]
    z_lim = (0, max(3*scale_it,range))

    dr1 = 0.4
    dr2 = 0.4



    if d_sim > d_real
        """
        1st Ball
        """
        plot!(fig, xlim = x_lim, ylim = y_lim, zlim = z_lim, ratio = :equal) # TODO zlim problematesch

        dx = xt[1]
        dy = xt[2]

        ϕ_y = xt[4]#*(pi/180)
        ϕ_x = -xt[5]#*(pi/180)
        ϕ_z = xt[6]#*(pi/180)




        radi = scale_it * 0.5

        fac_x = scale_it * 0.4
        fac_y = scale_it * 0.4
        fac_z = scale_it * 1

        dz   = 2 * radi + fac_z

        #Draw the Circle:
        u = 0.0:dr1:2pi+ dr1
        v = 0.0:dr2:pi+ dr2

        lu = length(u);
        lv = length(v);

        x = zeros(lu,lv);
        y = zeros(lu,lv);
        z = zeros(lu,lv);

        colors = zeros(lu,lv,3) ;

        colortube = 0.1:0.1:1.0;

        for uu=1:lu
            for vv=1:lv
                x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
                y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
                z[uu,vv]= radi*cos(v[vv]) + radi
            end
        end

        surface!(x,y,z, c=:blue);
        #surf(x,y,z)


        #Draw the Quader on Top
        x = -1:1:1
        y = -1:1:1

        lx = length(x)
        ly = length(y)

        X = zeros(lx,ly)
        Y = zeros(lx,ly)
        Z = zeros(lx,ly)

        for i_x = 1:lx
            for i_y = 1:ly
                X[i_x,i_y] = x[i_x]
                Y[i_x,i_y] = y[i_y]
                Z[i_x,i_y] = 1
            end
        end

        arr_X = [X,Y,Z,-Y,-Z,-Y]
        arr_Y = [Y,Z,X,-Z,-X,-Z]
        arr_Z = [Z,X,Y,-X,-Y,-Z]

        for i_side = 1:6
            #x-Koordinate Calculation
            x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
            x_koo .+= dx + sin(ϕ_y)*2*radi
            #y-Koordinate Calculation
            y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
            y_koo .+= dy + sin(ϕ_x)*2*radi
            #z-Koordinate Calculation
            z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
            z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

            surface!(fig,x_koo,y_koo,z_koo, c=:blue)
        end

        """
        2nd ball
        """

        dx = xt2[1]
        dy = xt2[2]

        ϕ_y = xt2[4]#*(pi/180)
        ϕ_x = -xt2[5]#*(pi/180)
        ϕ_z = xt2[6]#*(pi/180)




        radi = scale_it * 0.5

        fac_x = scale_it * 0.4
        fac_y = scale_it * 0.4
        fac_z = scale_it * 1

        dz   = 2 * radi + fac_z

        #Draw the Circle:
        u = 0.0:dr1:2pi+ dr1
        v = 0.0:dr2:pi+ dr2

        lu = length(u);
        lv = length(v);

        x = zeros(lu,lv);
        y = zeros(lu,lv);
        z = zeros(lu,lv);

        colors = zeros(lu,lv,3) ;

        colortube = 0.1:0.1:1.0;

        for uu=1:lu
            for vv=1:lv
                x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
                y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
                z[uu,vv]= radi*cos(v[vv]) + radi
            end
        end

        surface!(x,y,z, c=:green);
        #surf(x,y,z)


        #Draw the Quader on Top
        x = -1:1:1
        y = -1:1:1

        lx = length(x)
        ly = length(y)

        X = zeros(lx,ly)
        Y = zeros(lx,ly)
        Z = zeros(lx,ly)

        for i_x = 1:lx
            for i_y = 1:ly
                X[i_x,i_y] = x[i_x]
                Y[i_x,i_y] = y[i_y]
                Z[i_x,i_y] = 1
            end
        end

        arr_X = [X,Y,Z,-Y,-Z,-Y]
        arr_Y = [Y,Z,X,-Z,-X,-Z]
        arr_Z = [Z,X,Y,-X,-Y,-Z]

        for i_side = 1:6
            #x-Koordinate Calculation
            x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
            x_koo .+= dx + sin(ϕ_y)*2*radi
            #y-Koordinate Calculation
            y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
            y_koo .+= dy + sin(ϕ_x)*2*radi
            #z-Koordinate Calculation
            z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
            z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

            surface!(fig,x_koo,y_koo,z_koo, α = 0.5, c=:green)
        end

    else

        """
        2nd ball
        """

        plot!(fig, xlim = x_lim, ylim = y_lim, zlim = z_lim, ratio = :equal) # TODO zlim problematesch

        dx = xt2[1]
        dy = xt2[2]

        ϕ_y = xt2[4]#*(pi/180)
        ϕ_x = -xt2[5]#*(pi/180)
        ϕ_z = xt2[6]#*(pi/180)




        radi = scale_it * 0.5

        fac_x = scale_it * 0.4
        fac_y = scale_it * 0.4
        fac_z = scale_it * 1

        dz   = 2 * radi + fac_z

        #Draw the Circle:
        u = 0.0:dr1:2pi+ dr1
        v = 0.0:dr2:pi+ dr2

        lu = length(u);
        lv = length(v);

        x = zeros(lu,lv);
        y = zeros(lu,lv);
        z = zeros(lu,lv);

        colors = zeros(lu,lv,3) ;

        colortube = 0.1:0.1:1.0;

        for uu=1:lu
            for vv=1:lv
                x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
                y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
                z[uu,vv]= radi*cos(v[vv]) + radi
            end
        end

        #plot!(x,y,z);
        surface!(x,y,z, color = "blue");


        #Draw the Quader on Top
        x = -1:1:1
        y = -1:1:1

        lx = length(x)
        ly = length(y)

        X = zeros(lx,ly)
        Y = zeros(lx,ly)
        Z = zeros(lx,ly)

        for i_x = 1:lx
            for i_y = 1:ly
                X[i_x,i_y] = x[i_x]
                Y[i_x,i_y] = y[i_y]
                Z[i_x,i_y] = 1
            end
        end

        arr_X = [X,Y,Z,-Y,-Z,-Y]
        arr_Y = [Y,Z,X,-Z,-X,-Z]
        arr_Z = [Z,X,Y,-X,-Y,-Z]

        for i_side = 1:6
            #x-Koordinate Calculation
            x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
            x_koo .+= dx + sin(ϕ_y) * 2 * radi
            #y-Koordinate Calculation
            y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
            y_koo .+= dy + sin(ϕ_x)*2*radi
            #z-Koordinate Calculation
            z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
            z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

            surface!(fig,x_koo,y_koo,z_koo, α = 0.5, c=:green)
        end


        """
        1st ball
        """

        dx = xt[1]
        dy = xt[2]

        ϕ_y = xt[4]#*(pi/180)
        ϕ_x = -xt[5]#*(pi/180)
        ϕ_z = xt[6]#*(pi/180)




        radi = scale_it * 0.5

        fac_x = scale_it * 0.4
        fac_y = scale_it * 0.4
        fac_z = scale_it * 1

        dz   = 2 * radi + fac_z

        #Draw the Circle:
        u = 0.0:dr1:2pi+ dr1
        v = 0.0:dr2:pi+ dr2

        lu = length(u);
        lv = length(v);

        x = zeros(lu,lv);
        y = zeros(lu,lv);
        z = zeros(lu,lv);

        colors = zeros(lu,lv,3) ;

        colortube = 0.1:0.1:1.0;

        for uu=1:lu
            for vv=1:lv
                x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
                y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
                z[uu,vv]= radi*cos(v[vv]) + radi
            end
        end

        surface!(x,y,z, c=:blue);
        #surf(x,y,z)


        #Draw the Quader on Top
        x = -1:1:1
        y = -1:1:1

        lx = length(x)
        ly = length(y)

        X = zeros(lx,ly)
        Y = zeros(lx,ly)
        Z = zeros(lx,ly)

        for i_x = 1:lx
            for i_y = 1:ly
                X[i_x,i_y] = x[i_x]
                Y[i_x,i_y] = y[i_y]
                Z[i_x,i_y] = 1
            end
        end

        arr_X = [X,Y,Z,-Y,-Z,-Y]
        arr_Y = [Y,Z,X,-Z,-X,-Z]
        arr_Z = [Z,X,Y,-X,-Y,-Z]

        for i_side = 1:6
            #x-Koordinate Calculation
            x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
            x_koo .+= dx + sin(ϕ_y)*2*radi
            #y-Koordinate Calculation
            y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
            y_koo .+= dy + sin(ϕ_x)*2*radi
            #z-Koordinate Calculation
            z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
            z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

            surface!(fig,x_koo,y_koo,z_koo, c=:blue, α = 0.5)
        end

    end
    gui()

end

using JLD2
using FileIO

filename = "results_ballvsball.jld2"
a = load(filename)
data = a["(X, X2, u, u2, t, des_traj_real)"]
X  = data[1].parent'
X2 = data[2].parent'
u = data[3].parent'
u2 = data[4].parent'
t = data[5]
des_traj_real = data[6]

newfile = ("ballvsball.mp4")
visualise_ballbott_compare(X,X2,u,u2,t, newfile, des_traj_real; options= Dict())




using Plots

fig = plot(0,0)
pyplot()

scale_it= 0.25
xt2 = randn(12)
xt = randn(12)
dr1 = 0.2
dr2 = 0.2

dx = xt2[1]
dy = xt2[2]

ϕ_y = xt2[4]#*(pi/180)
ϕ_x = -xt2[5]#*(pi/180)
ϕ_z = xt2[6]#*(pi/180)




radi = scale_it * 0.5

fac_x = scale_it * 0.4
fac_y = scale_it * 0.4
fac_z = scale_it * 1

dz   = 2 * radi + fac_z

#Draw the Circle:
u = 0.0:dr1:2pi+ dr1
v = 0.0:dr2:pi+ dr2

lu = length(u);
lv = length(v);

x = zeros(lu,lv);
y = zeros(lu,lv);
z = zeros(lu,lv);

colors = zeros(lu,lv,3) ;

colortube = 0.1:0.1:1.0;

for uu=1:lu
    for vv=1:lv
        x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
        y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
        z[uu,vv]= radi*cos(v[vv]) + radi
    end
end

#plot!(x,y,z);
surface!(x,y,z, surfacecolor = z .*1 .+ 3, α= 0.5);
gui()

#Draw the Quader on Top
x = -1:1:1
y = -1:1:1

lx = length(x)
ly = length(y)

X = zeros(lx,ly)
Y = zeros(lx,ly)
Z = zeros(lx,ly)

for i_x = 1:lx
    for i_y = 1:ly
        X[i_x,i_y] = x[i_x]
        Y[i_x,i_y] = y[i_y]
        Z[i_x,i_y] = 1
    end
end

arr_X = [X,Y,Z,-Y,-Z,-Y]
arr_Y = [Y,Z,X,-Z,-X,-Z]
arr_Z = [Z,X,Y,-X,-Y,-Z]

for i_side = 1:6
    #x-Koordinate Calculation
    x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
    x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
    x_koo .+= dx + sin(ϕ_y) * 2 * radi
    #y-Koordinate Calculation
    y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
    y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
    y_koo .+= dy + sin(ϕ_x)*2*radi
    #z-Koordinate Calculation
    z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
    z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
    z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
    z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

    surface!(fig,x_koo,y_koo,z_koo, surfacecolor = z .* 1 .+3, α= 0.5)
end

gui()

        dx = xt[1]
        dy = xt[2]

        ϕ_y = xt[4]#*(pi/180)
        ϕ_x = -xt[5]#*(pi/180)
        ϕ_z = xt[6]#*(pi/180)




        radi = scale_it * 0.5

        fac_x = scale_it * 0.4
        fac_y = scale_it * 0.4
        fac_z = scale_it * 1

        dz   = 2 * radi + fac_z

        #Draw the Circle:
        u = 0.0:dr1:2pi+ dr1
        v = 0.0:dr2:pi+ dr2

        lu = length(u);
        lv = length(v);

        x = zeros(lu,lv);
        y = zeros(lu,lv);
        z = zeros(lu,lv);

        colors = zeros(lu,lv,3) ;

        colortube = 0.1:0.1:1.0;

        for uu=1:lu
            for vv=1:lv
                x[uu,vv]= radi*cos(u[uu])*sin(v[vv]) + dx
                y[uu,vv]= radi*sin(u[uu])*sin(v[vv]) + dy
                z[uu,vv]= radi*cos(v[vv]) + radi
            end
        end

        surface!(x,y,z, surfacecolor = z .* 2 .+1, α= 0.5)
        #surf(x,y,z)


        #Draw the Quader on Top
        x = -1:1:1
        y = -1:1:1

        lx = length(x)
        ly = length(y)

        X = zeros(lx,ly)
        Y = zeros(lx,ly)
        Z = zeros(lx,ly)

        for i_x = 1:lx
            for i_y = 1:ly
                X[i_x,i_y] = x[i_x]
                Y[i_x,i_y] = y[i_y]
                Z[i_x,i_y] = 1
            end
        end

        arr_X = [X,Y,Z,-Y,-Z,-Y]
        arr_Y = [Y,Z,X,-Z,-X,-Z]
        arr_Z = [Z,X,Y,-X,-Y,-Z]

        for i_side = 1:6
            #x-Koordinate Calculation
            x_koo  = cos(ϕ_y)*(cos(ϕ_z)*arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            x_koo .+= sin(ϕ_y)* arr_Z[i_side] * fac_z
            x_koo .+= dx + sin(ϕ_y)*2*radi
            #y-Koordinate Calculation
            y_koo  = cos(ϕ_x)*(cos(ϕ_z)*arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            y_koo .+= sin(ϕ_x)*arr_Z[i_side]* fac_z
            y_koo .+= dy + sin(ϕ_x)*2*radi
            #z-Koordinate Calculation
            z_koo  =                            (cos(ϕ_y)*cos(ϕ_x)*arr_Z[i_side]) * fac_z
            z_koo .-= sin(ϕ_y)*(cos(ϕ_z) * arr_X[i_side] + sin(ϕ_z)*arr_Y[i_side]) * fac_x
            z_koo .-= sin(ϕ_x)*(cos(ϕ_z) * arr_Y[i_side] - sin(ϕ_z)*arr_X[i_side]) * fac_y
            z_koo .+= dz - (2-cos(ϕ_x)-cos(ϕ_y))*2*radi

            surface!(fig,x_koo,y_koo,z_koo, surfacecolor = z .* 2 .+1, α= 0.5)
        end


    gui()
