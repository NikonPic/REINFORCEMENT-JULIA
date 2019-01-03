function segway_rk4(xt::Vector{Float64}, ut::AbstractVector, dt::Number, params::Vector)
    xt1 = copy(xt)
    segway_rk4!(xt1, ut, dt, params)
    return xt1
end


function segway_rk4!(xt::AbstractVector, ut::AbstractVector, dt::Number, params::Vector)
    temp = Array{Float64}(7)

    #get partial results dy1-dy4
    dy1 = dxdt_segway(xt, ut, params);
    temp .= dy1.*(dt/2.0) .+ xt;

    dy2 = dxdt_segway(temp, ut, params);
    temp .= dy2.*(dt/2.0) .+ xt;

    dy3 = dxdt_segway(temp, ut, params);
    temp .= dy3.*dt .+ xt;

    dy4 = dxdt_segway(temp, ut, params);

    # update xt
    @. temp = dt*(dy1+2.0*(dy2+dy3)+dy4)/6.0

    #Check if segway has fallen #TODO limits anpassen
    if -pi/2>=xt[4]+temp[4]
        xt[4] = -pi/2
		xt[5:7] = 0.0
    elseif pi/2<=xt[4]+temp[4]
        xt[4] = pi/2
		xt[5:7] = 0.0
    else
        # Limit angle to lie between 0 and 2*pi
        @. xt += temp
        #xt[3] = mod(xt[3], 2*pi)
    end
    return nothing
end


"""
Segway dynamics, taken from Pathak2005.
State vector is [x0, y0, phi, alpha, dalpha, v, dphi]
"""
function dxdt_segway(x::Vector{Float64}, u::AbstractVector, params::Vector)
    g = 9.81 #N/kg

    # Rename variables
    Mb = params[1]
    Mw = params[2]
    R = params[3]
    cz = params[4]
    b = params[5]
    Ixx = params[6]
    Iyy = params[7]
    Izz = params[8]
    Iwa = params[9]
    Iwd = params[10]

    phi = x[3]
    alpha = x[4]
    dalpha = x[5]
    v = x[6]
    dphi = x[7]

    # Rigid Body Dynamics
    Dalpha = Mb^2*cos(alpha)^2*cz^2*R^2+((-Mb^2-2*Mw*Mb)*cz^2-2*Iyy*Mw-Iyy*Mb)*R^2-2*Mb*cz^2*Iwa-2*Iyy*Iwa
    Galpha = (-Mb*cz^2+Izz-Ixx)*R^2*cos(alpha)^2 + (Mb*cz^2+Ixx+2*Iwd+2*b^2*Mw)*R^2+2*b^2*Iwa
    H = 1/2*Mb*R^2*Izz+Iwa*Izz-Mw*R^2*Ixx-Iwa*Ixx-Mb*cz^2*Mw*R^2 - Mb*cz^2*Iwa -1/2*Mb*R^2*Ixx+Mw*R^2*Izz
    Kalpha = (-4*Iyy*Mb*R^2*cz-3*R^2*Mb^2*cz^3+Mb*R^2*cz*(Ixx-Izz))*sin(alpha) + (Mb*R^2*cz*(Ixx-Izz)+R^2*Mb^2*cz^3)*sin(3*alpha)
    f21 = sin(2*alpha)*dphi^2*H/Dalpha + Mb^2*cz^2*R^2*sin(2*alpha)*dalpha^2/(2*Dalpha) + (-2*Mb^2*R^2*cz-4*Iwa*Mb*cz-4*Mw*R^2*Mb*cz)*g*sin(alpha)/(2*Dalpha)
    f22 = Kalpha*dphi^2+(Mb^2*cz^2*R^2*g*sin(2*alpha))/(2*Dalpha)+(-4*Iyy*Mb*R^2*cz-4*R^2*Mb^2*cz^3)*sin(alpha)*dalpha^2/(4*Dalpha)
    f23 = (-(Ixx-Izz)*R^2-Mb*cz^2*R^2)*sin(2*alpha)*dalpha*dphi/Galpha-sin(alpha)*R^2*Mb*cz*v*dphi/Galpha
    g21 = (u[1]+u[2])*(Mb*R^2+2*Mw*R^2+2*Iwa+Mb*cos(alpha)*cz*R)/Dalpha
    g22 = -(u[1]+u[2])*R*(Mb*cos(alpha)*cz*R+Iyy+Mb*cz^2)/Dalpha
    g23 = (u[1]-u[2])*R*b/Galpha

    xdot = Array{Float64}(7)
    xdot[1] = cos(phi)*v
    xdot[2] = sin(phi)*v
    xdot[3] = x[7]
    xdot[4] = x[5]
    xdot[5] = f21 + g21
    xdot[6] = f22 + g22
    xdot[7] = f23 + g23

    return xdot
end
