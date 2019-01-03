
function heavychain_plot!(fig, chain::Chain, xmin::Number, xmax::Number, t::Number)
  const chain_length = 1.178
  const n_links = 21-1
  const link_length = chain_length/n_links

  const basewidth = 0.12 #width of the car
  const baseheight = 0.08 #height of the car

  #plot
  plot!(fig, ratio=:equal, xlim = (xmin, xmax), ylim = (-chain_length*1.1, chain_length*0.05+baseheight/2), title = "t=$(round(t, 1))", lab = "", border=true, grid=false)

  #plot zero line
  plot!(fig, [xmin, xmax], [0.0, 0.0], c=:black, l=:dot, lab = "", grid = false)

  #plot base
  base(x) = Plots.Shape(x+[-basewidth/2, basewidth/2, basewidth/2, -basewidth/2, -basewidth/2], [-baseheight/2, -baseheight/2, baseheight/2, baseheight/2, -baseheight/2])
  plot!(fig, base(chain.q[1]), c=:orange, lab = "")

  #calculate chain points using the angles
  x = Array{Float64}(n_links+1)
  y = Array{Float64}(n_links+1)
  x[1] = chain.q[1]
  y[1] = 0.0
  for i = 2:length(x)
    x[i] = x[i-1]+link_length*sin(chain.q[i])
    y[i] = y[i-1]-link_length*cos(chain.q[i])
  end

  #plot chain
  plot!(fig, x, y, c = :black, lw = 2, marker = :circle, ms = 3.5, lab = "")

end
