#using SegwayRobot

function dynmodel!(x::AbstractVector, u::AbstractVector, dt::Number, nn_params)
    # Segway definition
    Mb = 1.76 # kg # Masse des Körpers
    Mw = 0.147 # kg # Masse der Räder
    R = 0.07 # m #Radius der Räder
    cz = 0.077 # m #Schwerpunkt des Körpers Höhe über Radachse
    b = 0.1985/2 # m #Half the distance between the wheels (OO_w in paper)
    Ixx = (0.166^2+0.21^2)*Mb/12 + Mb*cz^2 # kg m^2 #Inertia of the body
    Iyy = (0.072^2+0.21^2)*Mb/12 + Mb*cz^2 # kg m^2 #Inertia of the body
    Izz = (0.072^2+0.166^2)*Mb/12 # kg m^2 #Inertia of the body
    Iwa = Mw*R^2/2 # kg m^2 #Inertia of the wheels about their axis
    Iwd = Mw*b^2 # kg m^2 #Inertia of the wheels about their axis
    seg = Segway_unmutable(Mb, Mw, R, cz, b, Ixx, Iyy, Izz, Iwa, Iwd)

    # Evaluate Neural Net for the torque
    tau = eval_nn_segway(nn_params, vcat(x[4:7], u))

    # Evaluate segway dynamics
    segway_rk4!(x, tau, dt, seg)

    return x
end

@inline function eval_nn_segway(w, x)
    for i=1:2:length(w)-2
        x = tanh.(w[i] * x .+ w[i+1])
    end
    return w[end-1] * x .+ w[end]
end


"""
State vector is [x0, y0, phi, alpha, dalpha, v, dphi, ir, il]
"""
function segway_cost(xt::AbstractVector, u::AbstractVector)
    cost = (xt[4]^2 + 0.1*xt[5]^2
            + 10*xt[2]^2 + 10*xt[1]^2 # or squared here...
            + 0.1* xt[7]^2
            + 0.01*(u'u)) # punish for falling over

    reward = 0.0 #reward for moving along the x-axis

    if xt[4]^2 > 1 #check if lager than pi/2
        terminal = 1
        cost *= 1.2
    else
        terminal = 0
    end

    return cost-reward, terminal
end
