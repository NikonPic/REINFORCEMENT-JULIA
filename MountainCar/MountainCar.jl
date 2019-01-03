module MountainCar
# defines move_car! for many different types


function move_car!(x::Vector, u::Vector, dt::Number)
    x[2] += 0.001*u[1] - 0.0025*cos(3*x[1])

    # Correct bounds
    if (x[2] < -0.07)
        x[2] = -0.07
    elseif (x[2]>0.07)
        x[2] = 0.07
    end

    # Update position
    if x[1] >= 0.5
        x[2] = 0
        return x
    end

    x[1] += x[2]

    #check if the car reached a boundary
    if (x[1]<=-1.2)
        x[1] = -1.2;
        x[2] = 0.0;
    end
    return x
end

#"""
#x[1] position
#x[2] velocity
#"""
function cost_car(x::Vector, u::Vector)
    terminal = 0

    #cost = -(-0.01+0.0025/3.0*( sin(3.0*x[1])-sin(3.0*0.5) )); #continous reward
    cost = -10*(x[1]+0.5)^2 - 1*x[2]^2

    if (x[1]>=0.5)
        cost     = cost-100
        terminal = 1
    end

    return cost, terminal
end


rk4_func  = move_car!  # set standard Environment function
cost_func = cost_car # set standard Environment cost-function
s_init = [-pi/6 , 0.0]  #set initial position for environment
u_range = [1.0]

export rk4_func, cost_func, s_init, u_range

end
