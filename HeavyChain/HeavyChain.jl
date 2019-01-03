module HeavyChain

type Chain
  q::Array{Float64, 1} #generalised coordinates. Contains [x0, alpha_0, alpha_1, alpha_n], the pos of the cart and all link-angles
  u::Array{Float64, 1} #time derivative of q
  massmat::Array{Float64, 2}
  fvec::Array{Float64, 1}
end

function create_chain()
  q = zeros(21) #cart position and all angles are zero
  u = zeros(21) #all velocities are zero
  massmat = zeros(21, 21) #mass matrix, memory is overwritten at each forward simulation step
  fvec = zeros(21) #force vector, memory is overwritten at each forward simulation step
  return Chain(q, u, massmat, fvec)
end

export Chain, create_chain

include("src/simulate.jl")
export simulate!

#include("src/plot.jl")
#export heavychain_plot!

include("src/misc.jl")
export getendpoint

end
