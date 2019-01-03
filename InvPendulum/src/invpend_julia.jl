"""
x[1] is dx/dt (the velocity of the car)
x[2] is x (the position of the car)
x[3] is d phi/dt (the angular velocity of the pendulum)
x[4] is phi (the angle of the pendulum, 0[rad] is at the top)
u    is Vm (the input voltage for the motor)
"""
function inv_pendulum_rk4(x_t::Vector{T}, u, dt::Number) where {T}
    x_t1 = copy(x_t)
    inv_pendulum_rk4!(x_t1, u, dt)
    return x_t1
end

"""
x[1] is dx/dt (the velocity of the car)
x[2] is x (the position of the car)
x[3] is d phi/dt (the angular velocity of the pendulum)
x[4] is phi (the angle of the pendulum, 0[rad] is at the top)
u    is Vm (the input voltage for the motor)
"""
function inv_pendulum_rk4!(x_t::AbstractVector, u, dt::Number)

    temp = zeros(4)

    #get partial results dy1-dy4
    dy1 = dxdt_invpendulum(x_t, u);
    temp = dy1.*(dt/2.0) .+ x_t;

    dy2 = dxdt_invpendulum(temp, u);
    temp = dy2.*(dt/2.0) .+ x_t;

    dy3 = dxdt_invpendulum(temp, u);
    temp = dy3.*dt .+ x_t;

    dy4 = dxdt_invpendulum(temp, u);

    #update x
    x_t .+= dt.*(dy1.+2.0.*(dy2.+dy3).+dy4)./6.0;

    #x_t[4] = mod(x_t[4]+pi, 2*pi) - pi;
end

"""
x[1] is dx/dt (the velocity of the car)
x[2] is x (the position of the car)
x[3] is d phi/dt (the angular velocity of the pendulum)
x[4] is phi (the angle of the pendulum, 0[rad] is at the top)
u    is Vm (the input voltage for the motor)
"""
function dxdt_invpendulum(x::AbstractVector, u)
    # Parameters
    K_m = 0.00767;  # Gegeninduktionskoeffizient
    K_t = 0.00767;  # Motordrehmomentkonstante
    l_p = 0.3302;   # 0.1778;# Pendellaenge von Drehpunkt bis Schwerpunkt
    M_p = 0.23;     # Masse des Pendels
    M_c = 1.073;    # 0.7031;  # 0.94;# Masse des Wagens
    g = 9.81;
    J_p = 0.0079;    # Traegheitsmoment des Pendels bzgl. seines Schwerpunkts
    B_p = 0.00725;   # 0.0024;  # Viskoser Daempungskoeffizient aus Sciht der Pendelachse
    n_g = 1.0;       # Wirkungsgrad Planetengetriebe
    K_g = 3.7;       # Uerbersetzung Planetengetriebe
    n_m = 1.0;       # Wirkungsgrad des Motors
    R_m = 2.6;       # Motorwiderstand
    r_mp = 6.35e-3;  # Radius des Motorritzels
    B_eq = 5.4;      # viskoser Dämpfungskoeffizient

    F_c = -(n_g*K_g*K_g*n_m*K_t*K_m*x[1])/(R_m*r_mp*r_mp)+(n_g*K_g*n_m*K_t*u[1])/(R_m*r_mp);
    nen = (M_c +M_p)*J_p + M_c*M_p*l_p*l_p + M_p*M_p*l_p*l_p*sin(x[4])*sin(x[4]);

    xdot = zeros(4)
    xdot[1] = (-(J_p + M_p*(l_p*l_p))*B_eq*(x[1]) - ((M_p*M_p)*(l_p*l_p*l_p) + J_p*M_p*l_p)*((x[3])*(x[3]))*sin(x[4])- M_p*l_p*B_p*(x[3])*cos(x[4]) + (J_p + M_p*(l_p*l_p))*F_c + (M_p*M_p)*(l_p*l_p)*g*cos(x[4])*sin(x[4]))/nen;
    xdot[2] = x[1];
    xdot[3] = (-M_p*l_p*B_eq*(x[1])*cos(x[4]) -(M_p*M_p)*(l_p*l_p)*(x[3])*(x[3])*sin(x[4])*cos(x[4]) + (M_c + M_p)*M_p*g*l_p*sin(x[4])-(M_c + M_p)*B_p*(x[3]) + F_c*M_p*l_p*cos(x[4]))/nen;
    xdot[4] = x[3];

    return xdot
end

"""
x[1] is dx/dt (the velocity of the car)
x[2] is x (the position of the car)
x[3] is d phi/dt (the angular velocity of the pendulum)
x[4] is phi (the angle of the pendulum, 0[rad] is at the top)
u    is Vm (the input voltage for the motor)
"""
#Returns the received reward for a certain Environment
function inv_pendulum_cost(x::AbstractVector, u)
    #We want to reward the upright position of the pendulum and a position of x=0
    cost = (x[4]^2 + 0.1*x[3]^2 + 0.001*(u'u) + 0.1*x[2]^2)

    #Use the terminal statement to break the trajectory:
    if x[4]^2 > 2.47
        terminal = 1
        cost *= 100
    else
        terminal = 0
    end

    return cost, terminal
end


#ALTERNATIVE For Balancing:
"""
x[1] is dx/dt (the velocity of the car)
x[2] is x (the position of the car)
x[3] is d phi/dt (the angular velocity of the pendulum)
x[4] is phi (the angle of the pendulum, 0[rad] is at the top)
u    is Vm (the input voltage for the motor)
"""
#Returns the received reward for a certain Environment
function inv_pendulum_cost_swingup(x::AbstractVector, u)

    terminal = 0
    #Give reward according to inverse of pendulum
    cost = -5*cos(x[4]) + 0.2*x[2]^2 - 0.5*x[3]^2 + 0.1*(u'u)

    if abs(x[2]) > 5
        cost += 100
        terminal = 1
    end

    #if not fallen over ;)
    #if x[4]^2 < 2.47
    if cos(x[4]) > 0.5
        #cost = (x[4]^2 + 2*x[3]^2 + 0.01*(u'u) + 0.1*x[2]^2) - 50
        cost = (x[4]^2 + 2*x[3]^2 + 0.01*(u'u) + 0.1*x[2]^2) - 50
    end

    return cost,terminal
end
