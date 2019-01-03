function getendpoint(chain::Chain)
  const link_length = 1.178/20
  Out = chain.q[1]
  for i = 2:length(chain.q)
    Out += link_length*sin(chain.q[i])
  end
  return Out::Float64
end
