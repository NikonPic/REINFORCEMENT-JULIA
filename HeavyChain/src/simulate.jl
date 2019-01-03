# q contains the position fo the cart and all the angles of the links
# u are thze time derivatives of q | u=dq/dt
# F is the force on the cart
# dt is the time step size
# M and fvec are just placeholders that are being overwritten

include("massmat20.jl")
include("forcevec20.jl")
function simulate!(chain::Chain, F::Float64, dt::Float64)
            @inbounds begin
                        #assemble state vector
                        qu = vcat(chain.q, chain.u)
                        quf = vcat(qu, F)

                        #rk4
                        massmat20!(chain.massmat, quf)
                        forcevec20!(chain.fvec, quf)
                        dq1 = chain.u
                        du1 = chain.massmat\chain.fvec

                        quf[1:42] .= qu+dt/2.0*vcat(dq1, du1)
                        massmat20!(chain.massmat, quf)
                        forcevec20!(chain.fvec, quf)
                        dq2 = chain.u + dt/2.0*dq1
                        du2 = chain.massmat\chain.fvec

                        #vv_copy!(quf, vcat(qu+dt/2*vcat(dq2, du2), F))
                        quf[1:42] .= qu+dt/2.0*vcat(dq2, du2)
                        massmat20!(chain.massmat, quf)
                        forcevec20!(chain.fvec, quf)
                        dq3 = chain.u + dt/2.0*dq2
                        du3 = chain.massmat\chain.fvec

                        #vv_copy!(quf, vcat(qu+dt*vcat(dq3, du3), F))
                        quf[1:42] .= qu+dt*vcat(dq3, du3)
                        massmat20!(chain.massmat, quf)
                        forcevec20!(chain.fvec, quf)
                        dq4 = chain.u + dt*dq3
                        du4 = chain.massmat\chain.fvec

                        #update chain object
                        chain.q .+= dt/6 .* (dq1 .+ 2*dq2 .+ 2*dq3 .+ dq4)
                        chain.u .+= dt/6 .* (du1 .+ 2*du2 .+ 2*du3 .+ du4)
            end
end
