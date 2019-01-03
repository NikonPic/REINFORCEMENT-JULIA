#-------------------------------------------------------------------------------
#"""
#Containing all relevant parameters:
#"""
@everywhere mutable struct Params
    structure_net_vf ::Array{Int,1};
    structure_net_qf ::Array{Int,1};
    structure_net_qf2::Array{Int,1};
    structure_net_pl ::Array{Int,1};
    init_range       ::Float64;
    lr_pl            ::Float64;
    lr_vf            ::Float64;
    lr_qf            ::Float64;
    σ                ::Array{Float64,1};
    dt               ::Float64;
    t_horizon        ::Float64;
    batch_size       ::Int;
    l_traj           ::Int;
    n_traj           ::Int;
    n_iter           ::Int;
    γ_disc           ::Float64;
    u_range          ::Array{Float64,1};
    s_init           ::Array{Float64,1};
    b_t              ::Array{Float64,1};
    l_s_t            ::Int;
    l_out            ::Int;
    λ_critic         ::Float64;
    λ_actor          ::Float64;
    eps              ::Float64;
    δ                ::Float64;
    cg_damping       ::Float64;
    τ                ::Float64;
    buffer_size      ::Int;
    σ_off_policy     ::Float64;
    clip_σ           ::Float64;
    d_update         ::Int;
    Params() = new(); #to instanziate easily!
end
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#"""
#Containing the local saving of one gradient
#"""
@everywhere mutable struct Net
    grad  ::Array{Any,1}
    Net() = new()
end


"""
Initialize the Network
"""
function init_net(θ::Any)
    n_net = Net()
    n_net.grad = copy(θ) .* 0
    return n_net
end


"""
Copy the Network
"""
function copy_net(o_net::Net)
    c_net = Net()
    c_net.grad = copy(o_net.grad)
    return c_net
end
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#"""
#Containing all data of one trajectory
#"""
@everywhere mutable struct Trajectory
    #Struct of the Data collected in one Episode
    s_t    ::Array{Float64,2};   #states
    s_t2   ::Array{Float64,2};   #new_states
    a_t    ::Array{Float64,2};   #actions
    p_a_t  ::Array{Float64,2};   #Probability of actions
    r_t    ::Array{Float64,1};   #rewards
    val_t  ::Array{Float64,1};   #used value function results
    out_t  ::Array{Float64,2};   #NN-output
    A_t    ::Array{Float64,1};   #Local Advantage
    R_t    ::Array{Float64,1};   #Local discounted reward
    len    ::Int;                #Actual length of the Trajectory

    Trajectory() = new(); #for easy instanziation
end


#"""
#Create new Instance of trajectory
#"""
@everywhere function init_traj(params::Params,l_traj::Int)
    #Get from param struct
    n_traj = params.n_traj;
    l_s_t  = params.l_s_t;
    l_out  = params.l_out;

    #Preallocate required data
    new_traj          = Trajectory()
    new_traj.s_t      = zeros(l_traj,l_s_t)
    new_traj.s_t2     = zeros(l_traj,l_s_t)
    new_traj.a_t      = zeros(l_traj,l_out)
    new_traj.p_a_t    = zeros(l_traj,l_out)
    new_traj.r_t      = zeros(l_traj)
    new_traj.val_t    = zeros(l_traj)
    new_traj.out_t    = zeros(l_traj,l_out)
    new_traj.A_t      = zeros(l_traj)
    new_traj.R_t      = zeros(l_traj)
    new_traj.len      = 0

    return new_traj
end


#"""
#Copy the trajectory
#"""
@everywhere function copy_traj(o_traj::Trajectory,params::Params)
    #Get from param struct
    l_traj = params.l_traj;

    #start coping:
    c_traj          = Trajectory()
    c_traj.s_t      = copy(o_traj.s_t)
    c_traj.s_t2     = copy(o_traj.s_t2)
    c_traj.a_t      = copy(o_traj.a_t)
    c_traj.p_a_t    = copy(o_traj.p_a_t)
    c_traj.r_t      = copy(o_traj.r_t)
    c_traj.out_t    = copy(o_traj.out_t)
    c_traj.A_t      = copy(o_traj.A_t)
    c_traj.R_t      = copy(o_traj.R_t)
    c_traj.val_t    = copy(o_traj.val_t)
    c_traj.len      = copy(o_traj.len)

    return c_traj
end
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#"""
#Containing all data of all trajectories
#"""
@everywhere mutable struct Traj_Data
    traj      ::Array{Trajectory,1} #All data of Trajectories
    timesteps ::Int
    n_traj    ::Int

    Traj_Data() = new(); #for easy instanziation
end


"""
Initialize the trajectory data:
"""
function init_all_traj(params::Params)
    n_traj = params.n_traj;
    l_traj = params.l_traj;
    l_s_t  = params.l_s_t;

    #all_traj = Traj_Data(Array{Trajectory,1}(undef,n_traj),0,n_traj);
    all_traj = Traj_Data()
    all_traj.traj = Array{Trajectory,1}(undef,n_traj)
    all_traj.timesteps = 0
    all_traj.n_traj = n_traj

    #Create instance of trajectory:
    traj = init_traj(params,l_traj)

    #Fill the big struct with the small one:
    for i_traj = 1:n_traj
        all_traj.traj[i_traj] = copy_traj(traj,params)
    end

    return all_traj;
end

#-------------------------------------------------------------------------------


#Struct Running Mean_Square
#-------------------------------------------------------------------------------
@everywhere mutable struct RunMeanStd
    mean  ::Array{Float64,1}
    var   ::Array{Float64,1}
    std   ::Array{Float64,1}
    count ::Int
end

#Init the thing:
function init_rms(s_t::Array{Float64,1})
    len = length(s_t)
    rms = RunMeanStd(zeros(len) , zeros(len) .+1 , zeros(len) .+1 , 2)
    return rms
end
