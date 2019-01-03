function segway_rk4(xt::AbstractVector, ut::AbstractVector, dt::Number, seg::Segway = Segway_unmutable())
    xt1 = copy(xt)
    segway_rk4!(xt1, ut, dt, seg)
    return xt1
end


function segway_rk4!(xt::AbstractVector, ut::AbstractVector, dt::Number, seg::Segway = Segway_unmutable())
    temp = similar(xt)

    #get partial results dy1-dy4
    dy1 = dxdt_segway(xt, ut, seg);
    temp .= dy1.*(dt/2.0) .+ xt;

    dy2 = dxdt_segway(temp, ut, seg);
    temp .= dy2.*(dt/2.0) .+ xt;

    dy3 = dxdt_segway(temp, ut, seg);
    temp .= dy3.*dt .+ xt;

    dy4 = dxdt_segway(temp, ut, seg);

    # update xt
    @. temp = dt*(dy1+2.0*(dy2+dy3)+dy4)/6.0

    #Check if segway has fallen #TODO limits anpassen
    #if -pi/2>=xt[4]+temp[4]
    #    xt[4] = -pi/2
	#	xt[5:7] = 0.0
    #elseif pi/2<=xt[4]+temp[4]
    #    xt[4] = pi/2
	#	xt[5:7] = 0.0
    #else
        xt .+= temp
    #end
    return nothing
end


"""
Segway dynamics, taken from Pathak2005.
State vector is [x0, y0, phi, alpha, dalpha, v, dphi]
"""
@inline function dxdt_segway(x::AbstractVector, u::AbstractVector, seg::Segway = Segway_unmutable())
    g = 9.81 #N/kg

    # Rename variables
    phi = x[3]
    alpha = x[4]
    dalpha = x[5]
    v = x[6]
    dphi = x[7]

    # Rigid Body Dynamics
    Dalpha = seg.Mb^2*cos(alpha)^2*seg.cz^2*seg.R^2+((-seg.Mb^2-2*seg.Mw*seg.Mb)*seg.cz^2-2*seg.Iyy*seg.Mw-seg.Iyy*seg.Mb)*seg.R^2-2*seg.Mb*seg.cz^2*seg.Iwa-2*seg.Iyy*seg.Iwa
    Galpha = (-seg.Mb*seg.cz^2+seg.Izz-seg.Ixx)*seg.R^2*cos(alpha)^2 + (seg.Mb*seg.cz^2+seg.Ixx+2*seg.Iwd+2*seg.b^2*seg.Mw)*seg.R^2+2*seg.b^2*seg.Iwa
    H = 1/2*seg.Mb*seg.R^2*seg.Izz+seg.Iwa*seg.Izz-seg.Mw*seg.R^2*seg.Ixx-seg.Iwa*seg.Ixx-seg.Mb*seg.cz^2*seg.Mw*seg.R^2 - seg.Mb*seg.cz^2*seg.Iwa -1/2*seg.Mb*seg.R^2*seg.Ixx+seg.Mw*seg.R^2*seg.Izz
    Kalpha = (-4*seg.Iyy*seg.Mb*seg.R^2*seg.cz-3*seg.R^2*seg.Mb^2*seg.cz^3+seg.Mb*seg.R^2*seg.cz*(seg.Ixx-seg.Izz))*sin(alpha) + (seg.Mb*seg.R^2*seg.cz*(seg.Ixx-seg.Izz)+seg.R^2*seg.Mb^2*seg.cz^3)*sin(3*alpha)
    f21 = sin(2*alpha)*dphi^2*H/Dalpha + seg.Mb^2*seg.cz^2*seg.R^2*sin(2*alpha)*dalpha^2/(2*Dalpha) + (-2*seg.Mb^2*seg.R^2*seg.cz-4*seg.Iwa*seg.Mb*seg.cz-4*seg.Mw*seg.R^2*seg.Mb*seg.cz)*g*sin(alpha)/(2*Dalpha)
    f22 = Kalpha*dphi^2+(seg.Mb^2*seg.cz^2*seg.R^2*g*sin(2*alpha))/(2*Dalpha)+(-4*seg.Iyy*seg.Mb*seg.R^2*seg.cz-4*seg.R^2*seg.Mb^2*seg.cz^3)*sin(alpha)*dalpha^2/(4*Dalpha)
    f23 = (-(seg.Ixx-seg.Izz)*seg.R^2-seg.Mb*seg.cz^2*seg.R^2)*sin(2*alpha)*dalpha*dphi/Galpha-sin(alpha)*seg.R^2*seg.Mb*seg.cz*v*dphi/Galpha
    g21 = (u[1] + u[2])*(seg.Mb*seg.R^2+2*seg.Mw*seg.R^2+2*seg.Iwa+seg.Mb*cos(alpha)*seg.cz*seg.R)/Dalpha
    g22 = -(u[1] + u[2])*seg.R*(seg.Mb*cos(alpha)*seg.cz*seg.R+seg.Iyy+seg.Mb*seg.cz^2)/Dalpha
    g23 = (u[1] - u[2])*seg.R*seg.b/Galpha

    xdot = similar(x)
    xdot[1] = cos(phi)*v
    xdot[2] = sin(phi)*v
    xdot[3] = x[7]
    xdot[4] = x[5]
    xdot[5] = f21 + g21
    xdot[6] = f22 + g22
    xdot[7] = f23 + g23

    return xdot
end
