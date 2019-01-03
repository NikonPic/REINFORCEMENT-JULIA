function heavy_chain_plot!(fig, chain::Chain, xmin::Number, xmax::Number, t::Number)
  chain_length = chain.x_coords[end]
  const basewidth = 0.2 #width of the car
  const baseheight = 0.1 #height of the car

  #plot
  plot!(fig, ratio=:equal, xlim = (xmin, xmax), ylim = (-chain_length*1.1, chain_length*0.05+baseheight/2), title = "t=$(round(t, 1))", lab = "")

  #plot zero line
  plot!(fig, [xmin, xmax], [0.0, 0.0], c=:black, l=:dot, lab = "", grid = false)

  #plot base
  base(x) = Plots.Shape(x+[-basewidth/2, basewidth/2, basewidth/2, -basewidth/2, -basewidth/2], [-baseheight/2, -baseheight/2, baseheight/2, baseheight/2, -baseheight/2])
  plot!(fig, base(chain.w[1]), c=:orange, lab = "")

  #project x_coords to be consistent with the chain length
  x_new = Array{Float64}(length(chain.x_coords))
  x_new[1] = 0.0
  for i = 2:length(x_new)
    x_new[i] = x_new[i-1]+sqrt( (chain.x_coords[i]-chain.x_coords[i-1])^2 - (chain.w[i]-chain.w[i-1])^2 )
  end

  #plot chain
  plot!(fig, chain.w, -x_new, c = :black, lw = 2, marker = :circle, ms = 4, lab = "")
end
