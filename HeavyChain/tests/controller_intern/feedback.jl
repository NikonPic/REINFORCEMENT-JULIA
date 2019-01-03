#feedback controller (backstepping)
#inputs target is a 4 element vector containing: [x_target, v_target, phi0_target, phi0_dot_target]
function chain_feedback(chain::Chain , target_state::Array{Float64, 1}; alpha = 0.03, beta = 10, c = 10)

  #other parameters
  const g  = 9.81
  const rho = 2.3
  const L = 1.178
  const M = 6.1511

  #calculate difference between actual and desired state
  x_e = chain.q[1]-target_state[1]
  v_e = chain.u[1]-target_state[2]
  phi0_e = chain.q[2]-target_state[3]
  phi0_dot_e = chain.u[2]-target_state[4]

  #use it to generate output force
  r = -alpha*(-rho*g*L*phi0_e+c*x_e)
  r_dot = -alpha*(-rho*g*L*phi0_dot_e+c*v_e)
  ve_dot = r_dot + (1/alpha+beta)*r-beta*v_e
  F = M*ve_dot-rho*g*L*phi0_e

  return F

end
