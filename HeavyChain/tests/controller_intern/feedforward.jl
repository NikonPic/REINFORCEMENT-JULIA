function chain_feedforward(chain::Chain, all_t::Array{Float64, 1})
  const L = chain.x_coords[end]
  const rho = 2.3
  const g = 9.81
  const p0 = rho*g*L #mass of the chain
  const M = 6.1511 #mass of the car
  const h = 0.5 #how far the cart will drive

  #initialise outputs
  a0_target = 0.0 #acceleration at the top
  F_ff = Array{Float64}(length(all_t))
  xL_target = Array{Float64}(length(all_t)) #position of the bottom end of the chain (flacher Ausgang)
  x0_target = Array{Float64}(length(all_t)) #position of the car/top of the chain
  v0_target = Array{Float64}(length(all_t)) #velocity of the car/top of the chain
  phi0_target = Array{Float64}(length(all_t)) #chain angle at top of the chain
  phi0_dot_target = Array{Float64}(length(all_t)) #angular velocity of the chain at top of the chain

  poly1_coeffs, poly2_coeffs  = xL_poly_coeffs(h)

  #calculate (?flacher Ausgang?)
  for i = 1:length(all_t)
    xL_target[i] = y_w(all_t[i], poly1_coeffs, poly2_coeffs)
    x0_target[i] = x0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    v0_target[i] = v0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    phi0_target[i] = phi0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    phi0_dot_target[i] = dphi0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)

    F_ff[i] = M*a0_w(all_t[i], poly1_coeffs, poly2_coeffs, L) - p0*sin(phi0_target[i])
  end

  return F_ff, xL_target, x0_target, v0_target, phi0_target, phi0_dot_target
end

function chain_feedforward(all_t::Array{Float64, 1})
  const L = 1.178
  const rho = 2.3
  const g = 9.81
  const p0 = rho*g*L #mass of the chain
  const M = 6.1511 #mass of the car
  const h = 0.5 #how far the cart will drive

  #initialise outputs
  a0_target = 0.0 #acceleration at the top
  F_ff = Array{Float64}(length(all_t))
  xL_target = Array{Float64}(length(all_t)) #position of the bottom end of the chain (flacher Ausgang)
  x0_target = Array{Float64}(length(all_t)) #position of the car/top of the chain
  v0_target = Array{Float64}(length(all_t)) #velocity of the car/top of the chain
  phi0_target = Array{Float64}(length(all_t)) #chain angle at top of the chain
  phi0_dot_target = Array{Float64}(length(all_t)) #angular velocity of the chain at top of the chain

  poly1_coeffs, poly2_coeffs  = xL_poly_coeffs(h)

  #calculate (?flacher Ausgang?)
  for i = 1:length(all_t)
    xL_target[i] = y_w(all_t[i], poly1_coeffs, poly2_coeffs)
    x0_target[i] = x0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    v0_target[i] = v0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    phi0_target[i] = phi0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)
    phi0_dot_target[i] = dphi0_w(all_t[i], poly1_coeffs, poly2_coeffs, L)

    F_ff[i] = M*a0_w(all_t[i], poly1_coeffs, poly2_coeffs, L) - p0*sin(phi0_target[i])
  end

  return F_ff, xL_target, x0_target, v0_target, phi0_target, phi0_dot_target
end


#calculate polynomial coefficients for the (?flacher Ausgang?)
function xL_poly_coeffs(h)
  const t1 = 0.7
  const t2 = t1+0.3465
  const t3 = 3.0
  const t4 = t3+0.3465

  const c4 = -2*(t1+t2)
  const c3 = (t1^2+t2^2+4*t1*t2)
  const c2 = -2*t1*t2*(t1+t2)
  const c1 = (t1*t2)^2
  poly1(t) = 1/5*t^5 + 1/4*t^4*c4 + 1/3*t^3*c3 + 1/2*t^2*c2 + t*c1 #integrated polynomial

  const k4 = -2*(t3+t4)
  const k3 = (t3^2+t4^2+4*t3*t4)
  const k2 = -2*t3*t4*(t3+t4)
  const k1 = (t3*t4)^2
  poly2(t) = 1/5*t^5 + 1/4*t^4*k4 + 1/3*t^3*k3 + 1/2*t^2*k2 + t*k1

  Out1 = Array{Float64}(6)
  Out1[1] = -poly1(t1)/(poly1(t2)-poly1(t1))
  Out1[2] = c1/(poly1(t2)-poly1(t1))
  Out1[3] = 1/2*c2/(poly1(t2)-poly1(t1))
  Out1[4] = 1/3*c3/(poly1(t2)-poly1(t1))
  Out1[5] = 1/4*c4/(poly1(t2)-poly1(t1))
  Out1[6] = 1/5/(poly1(t2)-poly1(t1))

  Out2 = Array{Float64}(6)
  Out2[1] = 1.0+poly2(t3)/(poly2(t4)-poly2(t3))
  Out2[2] = -k1/(poly2(t4)-poly2(t3))
  Out2[3] = -1/2*k2/(poly2(t4)-poly2(t3))
  Out2[4] = -1/3*k3/(poly2(t4)-poly2(t3))
  Out2[5] = -1/4*k4/(poly2(t4)-poly2(t3))
  Out2[6] = -1/5/(poly2(t4)-poly2(t3))

  return Out1*h, Out2*h
end

#Solltrajektorie
function y_w(t, poly1_coeffs, poly2_coeffs)
  const t1 = 0.7
  const t2 = t1+0.3465
  const t3 = 3.0
  const t4 = t3+0.3465
  const t_period = 5.0

  poly1(t) = poly1_coeffs[1] + poly1_coeffs[2]*t + poly1_coeffs[3]*t^2+ poly1_coeffs[4]*t^3+ poly1_coeffs[5]*t^4 + poly1_coeffs[6]*t^5
  poly2(t) = poly2_coeffs[1] + poly2_coeffs[2]*t + poly2_coeffs[3]*t^2+ poly2_coeffs[4]*t^3+ poly2_coeffs[5]*t^4 + poly2_coeffs[6]*t^5

  t = mod(t, t_period)
  if t<=t1
    Out = poly1(t1)
  elseif t1<t<=t2
    Out = poly1(t)
  elseif t2<t<=t3
    Out = poly1(t2)
  elseif t3<t<=t4
    Out = poly2(t)
  else
    Out = poly2(t4)
  end
  return Out::Float64
end

function dydt_w(t, poly1_coeffs, poly2_coeffs)
  const t1 = 0.7
  const t2 = t1+0.3465
  const t3 = 3.0
  const t4 = t3+0.3465
  const t_period = 5.0

  poly1(t) = poly1_coeffs[2] + 2*poly1_coeffs[3]*t+ 3*poly1_coeffs[4]*t^2+ 4*poly1_coeffs[5]*t^3 + 5*poly1_coeffs[6]*t^4
  poly2(t) = poly2_coeffs[2] + 2*poly2_coeffs[3]*t+ 3*poly2_coeffs[4]*t^2+ 4*poly2_coeffs[5]*t^3 + 5*poly2_coeffs[6]*t^4

  t = mod(t, t_period)
  if t<=t1
    Out = 0.0
  elseif t1<t<=t2
    Out = poly1(t)
  elseif t2<t<=t3
    Out = 0.0
  elseif t3<t<=t4
    Out = poly2(t)
  else
    Out = 0.0
  end
  return Out::Float64
end

function ddydt2_w(t, poly1_coeffs, poly2_coeffs)
  const t1 = 0.7
  const t2 = t1+0.3465
  const t3 = 3.0
  const t4 = t3+0.3465
  const t_period = 5.0

  poly1(t) = 2*poly1_coeffs[3] + 6*poly1_coeffs[4]*t+ 12*poly1_coeffs[5]*t^2 + 20*poly1_coeffs[6]*t^3
  poly2(t) = 2*poly2_coeffs[3] + 6*poly2_coeffs[4]*t+ 12*poly2_coeffs[5]*t^2 + 20*poly2_coeffs[6]*t^3

  t = mod(t, t_period)
  if t<=t1
    Out = 0.0
  elseif t1<t<=t2
    Out = poly1(t)
  elseif t2<t<=t3
    Out = 0.0
  elseif t3<t<=t4
    Out = poly2(t)
  else
    Out = 0.0
  end
  return Out::Float64
end


function x0_w(t::Number, poly1_coeffs::Array{Float64, 1}, poly2_coeffs::Array{Float64, 1}, L::Number)
  const g = 9.81 #N/kg acceleration through gravity
  #return 1/pi*quadgk(x->y_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, abstol = 1e-8)[1] #integral
  return 1/pi*integral_simpson(x->y_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, 300) #integral
end


function v0_w(t::Number, poly1_coeffs::Array{Float64, 1}, poly2_coeffs::Array{Float64, 1}, L::Number)
  const g = 9.81 #N/kg acceleration through gravity
  #return 1/pi*quadgk(x->dydt_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, abstol = 1e-8)[1] #integral
  return 1/pi*integral_simpson(x->dydt_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, 300) #integral
end

function a0_w(t::Number, poly1_coeffs::Array{Float64, 1}, poly2_coeffs::Array{Float64, 1}, L::Number)
  const g = 9.81 #N/kg acceleration through gravity
  #return 1/pi*quadgk(x->ddydt2_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, abstol = 1e-8)[1] #integral
  return 1/pi*integral_simpson(x->ddydt2_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs), -pi/2, pi/2, 300) #integral
end

function phi0_w(t::Number, poly1_coeffs::Array{Float64, 1}, poly2_coeffs::Array{Float64, 1}, L::Number)
  const g = 9.81 #N/kg acceleration through gravity
  #return 1/pi*quadgk(x->(-dydt_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs)*sin(x)/(sqrt(L*g))), -pi/2, pi/2, abstol = 1e-8)[1] #integral
  return 1/pi*integral_simpson(x->(-dydt_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs)*sin(x)/(sqrt(L*g))), -pi/2, pi/2, 300) #integral
end

function dphi0_w(t::Number, poly1_coeffs::Array{Float64, 1}, poly2_coeffs::Array{Float64, 1}, L::Number)
  const g = 9.81 #N/kg acceleration through gravity
  #return 1/pi*quadgk(x->(-ddydt2_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs)*sin(x)/(sqrt(L*g))), -pi/2, pi/2, abstol = 1e-8)[1] #integral
  return 1/pi*integral_simpson(x->(-ddydt2_w(t+2*sqrt(L/g)*sin(x), poly1_coeffs, poly2_coeffs)*sin(x)/(sqrt(L*g))), -pi/2, pi/2, 300) #integral
end

#source: https://rosettacode.org/wiki/Numerical_integration
function integral_simpson(f::Function, xmin::Number, xmax::Number, N::Int)
  h = (xmax-xmin)/N
  sum1 = f(xmin+h/2)
  sum2 = 0.0
  for i = 1:(N-1)
    sum1 += f(xmin+h*(i+0.5))
    sum2 += f(xmin+h*i)
  end
  return h/6*(f(xmin)+f(xmax)+4*sum1+2*sum2)
end
