#Contains the functions for layer normalization! https://arxiv.org/abs/1607.06450

@everywhere using Statistics
if norm_active == true

    """
    Normalize
    """
    #Struct Normalized Activation
    #-------------------------------------------------------------------------------
    #Init the Net
    function init_θ_single_norm(net_s::Array{Int,1})
        x      = net_s[1]
        hidden = net_s[2:end-1]
        ysize  = net_s[end]

        w = []
        for y in [hidden...]
            push!(w, xavier(y, x))
            push!(w, ones(y))
            push!(w, zeros(y))
            x = y
        end
        push!(w, xavier(ysize, x)) # Prob actions
        push!(w, xavier(ysize, 1))

        return w
    end

    @everywhere function get_norm(x)
        x_mean = mean(x)
        x_std  = sqrt((x .- x_mean)' * (x .- x_mean) / length(x) + 1e-5)
        return x_mean, x_std
    end
    #@everywhere @zerograd get_norm(x)

    #Evaluate the Policy-Network
    @everywhere function mlp_nlayer_policy_norm(w::Any,x::AbstractVector,params::Params)
        σ       = params.σ
        u_range = params.u_range

        for i = 1:3:length(w)-2
            #1. Get the new summed input
            x = w[i] * x

            #2. Calculate the mean and std of the layer
            x_mean, x_std = get_norm(x)

            #3. Normalize activation by applying the norm and mean
            x_norm = (x .- x_mean) / x_std

            #4. Reparametrize the output
            x = x_norm .* w[i+1] + w[i+2]

            #5. Apply the non-linearity
            x = relu.(x)
        end

        mu_act   = (1 .- σ) .* u_range .* tanh.(w[end-1] * x .+ w[end])
        return vec(mu_act)
    end



    #Evaluate the Value-Network
    @everywhere function mlp_nlayer_value_norm(w::Any,x::AbstractVector,params::Params)
        eps = 1e-5

        for i = 1:3:length(w)-2
            #1. Get the new summed input
            x = w[i] * x

            #2. Calculate the mean and std of the layer
            x_mean, x_std = get_norm(x)

            #3. Normalize activation by applying the norm and mean
            x_norm = (x .- x_mean) / x_std

            #4. Reparametrize the output
            x = x_norm .* w[i+1] + w[i+2]

            #5. Apply the non-linearity
            x = relu.(x)
        end

        value    = (w[end-1] * x .+ w[end])[1]
        return value
    end

else

    """
    No normalization
    """

    init_θ_single_norm     = init_θ_single
    @everywhere mlp_nlayer_value_norm  = mlp_nlayer_value
    @everywhere mlp_nlayer_policy_norm = mlp_nlayer_policy

end
