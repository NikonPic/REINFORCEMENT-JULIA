abstract type Segway end

################################# Mutable version ###########################

mutable struct Segway_mutable<:Segway
    Mb::Float64 # kg # Masse des Körpers
    Mw::Float64 # kg # Masse der Räder
    R::Float64 # m #Radius der Räder
    cz::Float64 # m #Schwerpunkt des Körpers Höhe
    b::Float64 # m #Half the distance between the wheels (OO_w in paper)
    Ixx::Float64 # kg m^2 #Inertia of the body
    Iyy::Float64 # kg m^2 #Inertia of the body
    Izz::Float64 # kg m^2 #Inertia of the body
    Iwa::Float64 # kg m^2 #Inertia of the wheels about their axis
    Iwd::Float64 # kg m^2 #Inertia of the wheels about a diameter
end

function Segway_mutable(;
    Mb::Float64 = 1.76, # kg # Masse des Körpers
    Mw::Float64 = 0.147, # kg # Masse der Räder
    R::Float64 = 0.07, # m #Radius der Räder
    cz::Float64 = 0.077, # m #Schwerpunkt des Körpers Höhe über Radachse
    b::Float64 = 0.1985/2, # m #Half the distance between the wheels (OO_w in paper)
    Ixx::Float64 = (0.166^2+0.21^2)*Mb/12 + Mb*cz^2, # kg m^2 #Inertia of the body
    Iyy::Float64 = (0.072^2+0.21^2)*Mb/12 + Mb*cz^2, # kg m^2 #Inertia of the body
    Izz::Float64 = (0.072^2+0.166^2)*Mb/12, # kg m^2 #Inertia of the body
    Iwa::Float64 = Mw*R^2/2, # kg m^2 #Inertia of the wheels about their axis
    Iwd::Float64 = Mw*b^2) # kg m^2 #Inertia of the wheels about their axis)

    return Segway_mutable(Mb, Mw, R, cz, b, Ixx, Iyy, Izz, Iwa, Iwd)
end

function Segway_mutable(params::Vector{Float64}, x::Vector{Float64}=zeros(7))
    @assert(length(params) == 10)
    return Segway_mutable(params...)
end


################################# Unmutable version ###########################
struct Segway_unmutable<:Segway
    Mb::Float64 # kg # Masse des Körpers
    Mw::Float64 # kg # Masse der Räder
    R::Float64 # m #Radius der Räder
    cz::Float64 # m #Schwerpunkt des Körpers Höhe
    b::Float64 # m #Half the distance between the wheels (OO_w in paper)
    Ixx::Float64 # kg m^2 #Inertia of the body
    Iyy::Float64 # kg m^2 #Inertia of the body
    Izz::Float64 # kg m^2 #Inertia of the body
    Iwa::Float64 # kg m^2 #Inertia of the wheels about their axis
    Iwd::Float64 # kg m^2 #Inertia of the wheels about a diameter
end

function Segway_unmutable(;
    Mb::Float64 = 1.76, # kg # Masse des Körpers
    Mw::Float64 = 0.147, # kg # Masse der Räder
    R::Float64 = 0.07, # m #Radius der Räder
    cz::Float64 = 0.077, # m #Schwerpunkt des Körpers Höhe über Radachse
    b::Float64 = 0.1985/2, # m #Half the distance between the wheels (OO_w in paper)
    Ixx::Float64 = (0.166^2+0.21^2)*Mb/12 + Mb*cz^2, # kg m^2 #Inertia of the body
    Iyy::Float64 = (0.072^2+0.21^2)*Mb/12 + Mb*cz^2, # kg m^2 #Inertia of the body
    Izz::Float64 = (0.072^2+0.166^2)*Mb/12, # kg m^2 #Inertia of the body
    Iwa::Float64 = Mw*R^2/2, # kg m^2 #Inertia of the wheels about their axis
    Iwd::Float64 = Mw*b^2) # kg m^2 #Inertia of the wheels about their axis)

    return Segway_unmutable(Mb, Mw, R, cz, b, Ixx, Iyy, Izz, Iwa, Iwd)
end

function Segway_unmutable(params::Vector{Float64}, x::Vector{Float64}=zeros(7))
    @assert(length(params) == 10)
    return Segway_unmutable(params...)
end
