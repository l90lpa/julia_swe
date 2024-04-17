struct State{T <: AbstractFloat}
    u::Matrix{T}
    v::Matrix{T}
    h::Matrix{T}
end


function pad_field(f::Matrix{T}, geometry_padded::Geometry) where T <: AbstractFloat
    f_padded = zeros(get_locally_active_shape(geometry_padded))
    f_padded[at_local_domain(geometry_padded)] = f
    return f_padded
end


function pad_state(s::State{T}, geometry_padded::Geometry) where T <: AbstractFloat
    return State(pad_field(s.u, geometry_padded),
                 pad_field(s.v, geometry_padded),
                 pad_field(s.h, geometry_padded))
end


function unpad_field(f_padded::Matrix{T}, geometry_padded::Geometry) where T <: AbstractFloat
    return f_padded[at_local_domain(geometry_padded)]
end


function unpad_state(s_padded::State{T}, geometry_padded::Geometry) where T <: AbstractFloat
    return State(unpad_field(s_padded.u, geometry_padded),
                 unpad_field(s_padded.v, geometry_padded),
                 unpad_field(s_padded.h, geometry_padded))
end


function gather_global_state_domain(s::State{T}, geometry::Geometry, root::Int, comm) where T <: AbstractFloat
    s_local_domain = map(x -> x[at_local_domain(geometry)], values(s))
    s_global = map(x -> gather_matrix(x, (geometry.xproc, geometry.yproc), (geometry.nxprocs, geometry.nyprocs), comm, root), values(s_local_domain))
    return s_global
end