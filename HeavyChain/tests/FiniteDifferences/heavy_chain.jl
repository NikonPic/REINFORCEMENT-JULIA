type Chain
  x_coords::Array{Float64, 1} #space discretization, first entry must be 0, and then increasing order
  w::Array{Float64, 1} #deviation at x_coords
  w_dot::Array{Float64, 1} #velocities at x_coordinates

  A::AbstractMatrix
  B::AbstractVector
end


#constructor
function Chain(x_coords::Array{Float64, 1})
  @assert(x_coords[1] == 0.0)
  @assert length(x_coords)>3 #3 because 2nd derivatives will be needed
  @assert issorted(x_coords)
  if length(x_coords)<201
    A, B = hc_fdiff_mat(x_coords)
  else
    A, B = hc_fdiff_mat_sparse(x_coords)
  end
  return Chain(x_coords, zeros(length(x_coords)),  zeros(length(x_coords)), A, B)
end


function simulate!(chain::Chain, F::Number, dt::Float64)
  #some parameters
  N = length(chain.x_coords) #number of discrete points in space

  #simulate one step (Runge Kutta 4)
  v = vcat(chain.w_dot, chain.w)
  v1 = chain.A*v+chain.B*F
  v2 = chain.A*(v+dt/2*v1)+chain.B*F
  v3 = chain.A*(v+dt/2*v2)+chain.B*F
  v4 = chain.A*(v+dt*v3)+chain.B*F

  v += dt/6 * (v1 + 2*v2 + 2*v3 + v4)

  #update chain object
  chain.w_dot = v[1:N]
  chain.w = v[N+1:end]
end


function hc_fdiff_mat_old(x_coords)
  const rho = 2.3
  const g = 9.81
  const M = 6.151
  const L = x_coords[end]
  const N = length(x_coords)
  p(x::Number) = rho*g*(L-x)

  #preallocate output
  OutA = zeros(2*N, 2*N)
  OutB = vcat(1/M, zeros(2*N-1))

  #add boundary condition of the car
  OutA[1, N+1] = -p(x_coords[1])/(M*(x_coords[2]-x_coords[1]))
  OutA[1, N+2] = p(x_coords[1])/(M*(x_coords[2]-x_coords[1]))

  #add inner equations
  for row = 2:N-1
    OutA[row, N+row-1] = ( p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) - (p(x_coords[row+1])-p(x_coords[row-1]))/(x_coords[row+1]-x_coords[row-1]) )/rho

    OutA[row, N+row] = (-2*p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) )/rho

    OutA[row, N+row+1] = (p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) + (p(x_coords[row+1])-p(x_coords[row-1]))/(x_coords[row+1]-x_coords[row-1]) )/rho
  end

  #add boundary (double backwards Euler)
  row = N
  OutA[row, N+row-2] = ( p(x_coords[row-1])/((x_coords[row]-x_coords[row-1])*(x_coords[row-1]-x_coords[row-2])) )/rho
  OutA[row, N+row-1] = ( -p(x_coords[row-1])/((x_coords[row]-x_coords[row-1])*(x_coords[row-1]-x_coords[row-2])) - p(x_coords[row])/(x_coords[row]-x_coords[row-1])^2 )/rho
  OutA[row, N+row] = ( p(x_coords[row])/(x_coords[row]-x_coords[row-1])^2 )/rho

  #add I to the bottom left
  for row = 1:N
    OutA[row+N, row] = 1.0
  end

  return OutA, OutB
end

function hc_fdiff_mat(x_coords)
  const rho = 2.3 #almost not needed, got rid of it analytically
  const g = 9.81
  const M = 6.151
  const L = x_coords[end]
  const N = length(x_coords)
  p(x::Number) = g*(L-x)

  #preallocate output
  OutA = zeros(2*N, 2*N)
  OutB = vcat(1/M, zeros(2*N-1))

  #add boundary condition of the car
  OutA[1, N+1] = -p(x_coords[1])*rho/(M*(x_coords[2]-x_coords[1]))
  OutA[1, N+2] = p(x_coords[1])*rho/(M*(x_coords[2]-x_coords[1]))

  #add inner equations
  for row = 2:N-1
    OutA[row, N+row-1] = -g/(x_coords[row+1]-x_coords[row-1]) + p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))

    OutA[row, N+row] = -2*p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))

    OutA[row, N+row+1] = g/(x_coords[row+1]-x_coords[row-1]) + p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))
  end

  #add boundary (same as row above)
  row = N-1
  OutA[row+1, N+row-1] = -g/(x_coords[row+1]-x_coords[row-1]) + p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))
  OutA[row+1, N+row] = -2*p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))
  OutA[row+1, N+row+1] = g/(x_coords[row+1]-x_coords[row-1]) + p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1]))

  #row = N
  #OutA[row, N+row-1] = g/(x_coords[row]-x_coords[row-1])
  #OutA[row, N+row] = -g/(x_coords[row]-x_coords[row-1])

  #add I to the bottom left
  for row = 1:N-1
    OutA[row+N, row] = 1.0
  end
  OutA[2*N, N-1] = 1.0

  return OutA, OutB
end


function hc_fdiff_mat_sparse_old(x_coords)
  const rho = 2.3
  const g = 9.81
  const M = 6.151
  const L = x_coords[end]
  const N = length(x_coords)
  p(x::Number) = rho*g*(L-x)

  #preallocate output
  n_vals = N*4-1 #N*3-1+N
  I = Array{Int}(n_vals)
  J = Array{Int}(n_vals)
  Val = Array{Float64}(n_vals)

  #add boundary condition of the car
  I[1] = 1
  J[1] = N+1
  Val[1] = -p(x_coords[1])/(M*(x_coords[2]-x_coords[1]))

  I[2] = 1
  J[2] = N+2
  Val[2] = p(x_coords[1])/(M*(x_coords[2]-x_coords[1]))

  #add inner equations
  index = 3
  for row = 2:N-1
    I[index] = row
    J[index] = N+row-1
    Val[index] = ( p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) - (p(x_coords[row+1])-p(x_coords[row-1]))/(x_coords[row+1]-x_coords[row-1]) )/rho
    index += 1

    I[index] = row
    J[index] = N+row
    Val[index] = (-2*p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) )/rho
    index += 1

    I[index] = row
    J[index] = N+row+1
    Val[index] =  (p(x_coords[row])/((x_coords[row+1]-x_coords[row])*(x_coords[row]-x_coords[row-1])) + (p(x_coords[row+1])-p(x_coords[row-1]))/(x_coords[row+1]-x_coords[row-1]) )/rho
    index += 1
  end

  #add boundary (double backwards Euler)
  I[index] = N
  J[index] = 2*N-2
  Val[index] = ( p(x_coords[N-1])/((x_coords[N]-x_coords[N-1])*(x_coords[N-1]-x_coords[N-2])) )/rho
  index += 1

  I[index] = N
  J[index] = 2*N-1
  Val[index] = ( -p(x_coords[N-1])/((x_coords[N]-x_coords[N-1])*(x_coords[N-1]-x_coords[N-2])) - p(x_coords[N])/(x_coords[N]-x_coords[N-1])^2 )/rho
  index += 1

  I[index] = N
  J[index] = 2*N
  Val[index] = ( p(x_coords[N])/(x_coords[N]-x_coords[N-1])^2 )/rho
  index += 1

  #add I to the bottom left
  for row = 1:N
    I[index] = N+row
    J[index] = row
    Val[index] = 1.0
    index += 1
  end

  return sparse(I, J, Val, 2*N, 2*N), sparsevec([1], [1/M], 2*N)
end


function hc_fdiff_mat_sparse(x_coords)
  return hc_fdiff_mat_sparse_old(x_coords)
end
