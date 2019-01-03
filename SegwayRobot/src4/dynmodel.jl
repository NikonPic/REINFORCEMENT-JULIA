function dynmodel!(xt::AbstractVector, ut::AbstractVector, params::AbstractVector)
    ## Parameters and constants
    dt = 0.01
    # motor
    k_motor = exp(params[1])
    # k1_left = exp(params[1]) # Motor constant: current -> torque
    # k1_right = exp(params[2]) # Motor constant: current -> torque
    # segway
    Mb = 1.76 # kg # Masse des Körpers
    Mw = 0.147 # kg # Masse der Räder
    R = 0.07 # m #Radius der Räder
    cz = 0.077 # m #Schwerpunkt des Körpers Höhe über Radachse
    b = 0.1985/2 # m #Half the distance between the wheels (OO_w in paper)
    Ixx = (0.166^2+0.21^2)*Mb/12 + Mb*(cz-0.07)^2 # kg m^2 #Inertia of the body
    Iyy = (0.072^2+0.21^2)*Mb/12 + Mb*(cz-0.07)^2 # kg m^2 #Inertia of the body
    Izz = (0.072^2+0.166^2)*Mb/12 # kg m^2 #Inertia of the body
    Iwa = Mw*R^2/2 # kg m^2 #Inertia of the wheels about their axis
    Iwd = Mw*b^2 # kg m^2 #Inertia of the wheels about their axis
    seg = Segway_unmutable(Mb, Mw, R, cz, b, Ixx, Iyy, Izz, Iwa, Iwd)

    # friction
    fl_1 = exp(params[2]) # left wheel
    fr_1 = exp(params[3]) # right wheel
    fl_2 = exp(params[4]) # left wheel
    fr_2 = exp(params[5]) # right wheel

    ## Evaluate dynamics
    # get wheel rotation speeds
    v_left = xt[6]-xt[7]*b
    v_right = xt[6]+xt[7]*b
    omega_right = v_right/R - xt[5]
    omega_left = v_left/R - xt[5]
    # apply simplified motor model (omega taken out in favor of viscose friction)
    tau_right = k_motor*ut[2]
    tau_left = k_motor*ut[1]
    # apply simplified friction model
    tau_right = frictionmodel(tau_right, omega_right, fr_1, fr_2)
    tau_left = frictionmodel(tau_left, omega_left, fl_1, fl_2)
    # apply segway model
    tau = [tau_right, tau_left]
    segway_rk4!(xt, tau, dt, seg)

    return xt
end

@inline function frictionmodel(tau, omega, param1, param2)
    # Viskoser Reibterm
    dtau_v = 0.0

    # Haftreibung (geglättet)
    dtau_h = tau # Case tau < Fs and no rotation
    if (abs(omega) < 1e-3) && (param2 < tau) # Case tau > Fs and no rotation
        dtau_h = sign(tau)*param2
    elseif (abs(omega) > 1e-3)
        dtau_h = sign(omega)*param2
        dtau_v = param1*omega
    end

    return tau-dtau_v-dtau_h
end

# Does not overwrite x
function dynmodel(x::AbstractVector, u::AbstractVector, params::AbstractVector)
    x2 = copy(x)
    dynmodel!(x2, u, params)
    return x2
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

    if sum(isnan, xt) > 0
        terminal = 1
    end

    return cost-reward, terminal
end
