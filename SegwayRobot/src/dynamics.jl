function segway_rk4(xt::AbstractVector, ut::AbstractVector, dt::Number)
    xt1 = copy(xt)
    segway_rk4!(xt1, ut, dt)
    return xt1
end


function segway_rk4!(xt::AbstractVector, ut::AbstractVector, dt::Number)
    temp = similar(xt)

    #get partial results dy1-dy4
    dy1 = dxdt_segway(xt, ut);
    temp .= dy1.*(dt/2.0) .+ xt;

    dy2 = dxdt_segway(temp, ut);
    temp .= dy2.*(dt/2.0) .+ xt;

    dy3 = dxdt_segway(temp, ut);
    temp .= dy3.*dt .+ xt;

    dy4 = dxdt_segway(temp, ut);

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
        # Limit angle to lie between 0 and 2*pi
    @. xt += temp
        #xt[3] = mod(xt[3], 2*pi)
    #end
    return nothing
end


"""
State vector is [x0, y0, phi, alpha, dalpha, v, dphi, ir, il]
"""
function dxdt_segway(x::AbstractVector, u::AbstractVector)

    u1 = u[1];
    u2 = u[2];
    x3 = x[3];
    x4 = x[4];
    x5 = x[5];
    x6 = x[6];
    x7 = x[7];
    t2 = cos(x4);
    t3 = x6.*-2.424242424242424e2;
    t4 = t2.*1.368588515497592e-6;
    t5 = t2.^2;
    t6 = t5.*-1.979293098627009e-7;
    t7 = t4+t6+2.760783558017359e-6;
    t8 = 1.0./t7;
    t9 = x5.*8.0;
    t63 = x7.*1.187878787878788e1;
    t56 = -t63;
    t20 = -t56;
    t10 = -t20;
    t11 = t3+t9+t10;
    t12 = exp(t11);
    t13 = t12+1.0;
    t14 = 1.0./t13;
    t15 = t14.*6.519999999999999e-2;
    t16 = t2.*4.4489247e-4;
    t17 = t16-1.53811157502575e-3;
    t18 = x7.*2.274787878787879e-3;
    t19 = x5.*8.0;
    t21 = t16+3.788361628429753e-4;
    t22 = x5.*6.238865454545455e-3;
    t23 = x6.*-1.890565289256198e-1;
    t24 = cos(x3);
    t25 = sin(x3);
    t26 = sin(x4);
    t27 = x7.^2;
    t28 = x5.^2;
    t61 = x5.*1.532e-3;
    t54 = -t61;
    t31 = -t54;
    t29 = -t31;
    t62 = x6.*-4.642424242424242e-2;
    t55 = -t62;
    t32 = -t55;
    t30 = -t32;
    t33 = x6.*-2.424242424242424e2;
    t34 = u1.*3.3e-2;
    t65 = x7.*9.263769917355372e-3;
    t64 = -t65;
    t39 = -t64;
    t35 = -t39;
    t36 = t22+t23+t34+t35;
    t37 = t16+1.136228822457025e-3;
    t38 = u2.*3.3e-2;
    t40 = t2.*1.368588515497592e-6;
    t41 = t5.*-1.979293098627009e-7;
    t42 = t40+t41+2.760783558017359e-6;
    t43 = 1.0./t42;
    t44 = x5.*-3.064e-3;
    t45 = x6.*9.284848484848484e-2;
    t46 = t26.*1.322543979e-1;
    t53 = x7.*1.187878787878788e1;
    t47 = -t53;
    t48 = t19+t33+t47;
    t49 = exp(t48);
    t50 = t49+1.0;
    t51 = 1.0./t50;
    t52 = t51.*6.519999999999999e-2;
    t57 = t26.^2;
    t58 = t57.*1.1383382179917e-6;
    t59 = t58+4.056771650622809e-6;
    t60 = 1.0./t59;
    t66 = x4.*2.0;
    t67 = sin(t66);
    xDot = [t24.*x6;t25.*x6;x7;x5;t8.*(t15+t44+t45+t46+t27.*t67.*5.2265299265e-4+6.519999999999999e-2./(exp(t3+t19+x7.*1.187878787878788e1)+1.0)-6.519999999999999e-2).*1.916947737868725e-3-t8.*t21.*(t22+t23+t38+x7.*9.263769917355372e-3).*3.819323816679188-t8.*t17.*(t18+x5.*1.532e-3-x6.*4.642424242424242e-2-6.519999999999999e-2./(exp(t3+t19+t20)+1.0)+3.26e-2)+t8.*t17.*(t15+t18+t29+t30-3.26e-2)-t8.*t21.*t36.*3.819323816679188-t8.*t17.*t24.*(t24.*t26.*t27.*4.867e-2+t24.*t26.*t28.*4.867e-2+t25.*x6.*x7+t2.*t25.*x5.*x7.*9.734e-2).*9.141000000000001e-3-t8.*t17.*t25.*(t25.*t26.*t27.*4.867e-2+t25.*t26.*t28.*4.867e-2-t24.*x6.*x7-t2.*t24.*x5.*x7.*9.734e-2).*9.141000000000001e-3;t8.*(t18+t31+t32-6.519999999999999e-2./(exp(t19+t33+x7.*1.187878787878788e1)+1.0)+3.26e-2).*8.825323311693156e-5-t8.*(t18+t29+t30+t52-3.26e-2).*8.825323311693156e-5-t8.*t17.*(t44+t45+t46+t52+6.519999999999999e-2./(exp(t19+t33+t53)+1.0)+t2.*t26.*t27.*1.0453059853e-3-6.519999999999999e-2).*3.3e-2+t8.*t37.*(t22+t23+t38+t39).*1.260376859504132e-1+t8.*t36.*t37.*1.260376859504132e-1+t26.*t27.*t43.*3.926319886687748e-8+t26.*t28.*t43.*3.926319886687748e-8;t60.*(t22+t23+t34+t64).*6.175846611570248e-3-t60.*(t22+t23+t38+t65).*6.175846611570248e-3-t60.*(t18+t54+t55+6.519999999999999e-2./(exp(t3+t9+t56)+1.0)-3.26e-2).*1.617e-3-t60.*(t18+t61+t62-6.519999999999999e-2./(exp(t3+t19+t63)+1.0)+3.26e-2).*1.617e-3-t60.*x7.*(t26.*x6.*1.348159e-2+t67.*x5.*1.0453059853e-3).*1.089e-3];

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

    if xt[4]^2 > 2.47 #check if lager than pi/2
        terminal = 1
        cost *= 1.2
    else
        terminal = 0
    end

    return cost-reward, terminal
end

"""
State vector is [x0, y0, phi, alpha, dalpha, v, dphi, ir, il]
"""
function segway_cost2(xt::AbstractVector, u::AbstractVector)
    cost =  10.0*xt[1]^2 + 10.0*xt[2]^2
            + sin(xt[3])^2 + (1.0-cos(xt[3]))^2
            + 10*sin(xt[4])^2 + (1.0-cos(xt[4]))^2
            + 1*xt[5]^2 + 0.1*xt[6]^2
            + 0.1*xt[7]^2 + 0.1*u[1]^2
            + 0.1*u[2]^2

    if xt[4]^2 > 2.47 #check if lager than pi/2
        terminal = 1
        cost *= 10
    else
        terminal = 0
    end

    return cost, terminal
end
