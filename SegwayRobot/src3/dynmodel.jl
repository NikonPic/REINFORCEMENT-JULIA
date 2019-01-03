function dynmodel!(x::Vector, u::Vector, dt::Number, params::Vector)
    ## Parameters and constants
    # motor
    k1_left = exp(params[1]) # Motor constant: current -> torque
    k1_right = exp(params[2]) # Motor constant: current -> torque
    k2_left = exp(params[3]) # Motor constant: rot.speed -> induced voltage
    k2_right = exp(params[4]) # Motor constant: rot.speed -> induced voltage
    # segway
    Mb = 1.76 # kg # Masse des Körpers
    Mw = 0.147 # kg # Masse der Räder
    R = 0.07 # m #Radius der Räder
    cz = 0.077 # m #Schwerpunkt des Körpers Höhe über Radachse
    b = 0.1985/2 # m #Half the distance between the wheels (OO_w in paper)
    Ixx = (0.166^2+0.21^2)*Mb/12 + Mb*(cz-0.07)^2 # kg m^2 #Inertia of the body
    Iyy = (0.072^2+0.21^2)*Mb/12 + Mb*cz^2 # kg m^2 #Inertia of the body
    Izz = (0.072^2+0.166^2)*Mb/12 # kg m^2 #Inertia of the body
    Iwa = Mw*R^2/2 # kg m^2 #Inertia of the wheels about their axis
    Iwd = Mw*b^2 # kg m^2 #Inertia of the wheels about their axis
    seg = Segway_unmutable(Mb, Mw, R, cz, b, Ixx, Iyy, Izz, Iwa, Iwd)
    # friction
    fl_1 = exp(params[5]) # left wheel
    fl_2 = exp(params[6]) # left wheel
    fr_1 = exp(params[7]) # right wheel
    fr_2 = exp(params[8]) # right wheel

    ## Evaluate dynamics
    # get wheel rotation speeds
    v_left = x[6]-x[7]*b
    v_right = x[6]+x[7]*b
    omega_right = v_right/R
    omega_left = v_left/R
    # apply simplified motor model
    tau_right = k1_right*(u[2]-omega_right*k2_right)
    tau_left = k1_left*(u[1]-omega_left*k2_left)
    # apply simplified friction model
    tau_right = frictionmodel(tau_right, omega_right, fr_1, fr_2)
    tau_left = frictionmodel(tau_left, omega_left, fl_1, fl_2)
    # apply segway model
    tau = [tau_right, tau_left]
    segway_rk4!(x, tau, dt, seg)

    return x
end

function dynmodel(x::Vector, u::Vector, dt::Number, params::Vector)
    x2 = copy(x)
    dynmodel!(x2, u, dt, params)
    return x2
end

@inline function frictionmodel(tau, omega, param1, param2)
    # Viskoser Reibterm
    dtau_v = param1*omega

    # Haftreibung
    dtau_h = param2*sign(omega)

    return tau-dtau_v-dtau_h
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

    if xt[4]^2 > 2 #check if lager than pi/2
        terminal = 1
        cost *= 1.2
    else
        terminal = 0
    end

    return cost-reward, terminal
end
