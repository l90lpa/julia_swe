

function apply_model(s_new::State{T}, s::State{T}, geometry::Geometry, b::Matrix{T}, dt::T, dx::T, dy::T) where T <: AbstractFloat

    u = s.u
    v = s.v
    h = s.h

    dtdx = dt / dx
    dtdy = dt / dy
    g = 9.81

    (i, j) = at_local_domain(geometry)
    i_plus_1 = i.start+1 : i.stop+1
    j_plus_1 = j.start+1 : j.stop+1
    i_minus_1 = i.start-1 : i.stop-1
    j_minus_1 = j.start-1 : j.stop-1
    
    u_new_interior = ((u[i_plus_1, j] .+ u[i_minus_1, j] .+ u[i, j_plus_1] .+ u[i, j_minus_1]) ./ 4.0 
        .-0.5 .* dtdx .* ((u[i_plus_1, j].^2) ./ 2.0 .- (u[i_minus_1, j].^2) ./ 2.0)
        .-0.5 .* dtdy .* v[i, j] .* (u[i, j_plus_1] .- u[i, j_minus_1])
        .-0.5 .* dtdx .* g .* (h[i_plus_1, j] .- h[i_minus_1, j]))

    v_new_interior = ((v[i_plus_1, j] .+ v[i_minus_1, j] .+ v[i, j_plus_1] .+ v[i, j_minus_1]) ./ 4.0 
        .-0.5 .* dtdy .* ((v[i, j_plus_1].^2) ./ 2.0 .- (v[i, j_minus_1].^2) ./ 2.0)
        .-0.5 .* dtdx .* u[i, j] .* (v[i_plus_1, j] .- v[i_minus_1, j])
        .-0.5 .* dtdy .* g .* (h[i, j_plus_1] .- h[i, j_minus_1]))

    h_new_interior = ((h[i_plus_1, j] .+ h[i_minus_1, j] .+ h[i, j_plus_1] .+ h[i, j_minus_1]) ./ 4.0 
        .-0.5 .* dtdx .* u[i,j] .* ((h[i_plus_1, j] .- b[i_plus_1, j]) .- (h[i_minus_1, j] .- b[i_minus_1, j]))
        .-0.5 .* dtdy .* v[i,j] .* ((h[i, j_plus_1] .- b[i, j_plus_1]) .- (h[i, j_minus_1] .- b[i, j_minus_1]))
        .-0.5 .* dtdx .* (h[i,j] .- b[i,j]) .* (u[i_plus_1, j] .- u[i_minus_1, j])
        .-0.5 .* dtdy .* (h[i,j] .- b[i,j]) .* (v[i, j_plus_1] .- v[i, j_minus_1]))
    
    s_new.u[i, j] = u_new_interior
    s_new.v[i, j] = v_new_interior
    s_new.h[i, j] = h_new_interior

    return s_new
end


function apply_boundary_conditions(s_new::State{T}, s::State{T}, geometry::Geometry) where T <: AbstractFloat

    start, stop = get_locally_owned_range(geometry)
    x_slice = start.x : stop.x
    y_slice = start.y : stop.y

    if geometry.local_topology.south == -1
        s_new.u[x_slice, start.y] =  s.u[x_slice, start.y + 1]
        s_new.v[x_slice, start.y] = -s.v[x_slice, start.y + 1]
        s_new.h[x_slice, start.y] =  s.h[x_slice, start.y + 1]
    end

    if geometry.local_topology.north == -1
        s_new.u[x_slice, stop.y] =  s.u[x_slice, stop.y - 1]
        s_new.v[x_slice, stop.y] = -s.v[x_slice, stop.y - 1]
        s_new.h[x_slice, stop.y] =  s.h[x_slice, stop.y - 1]
    end

    if geometry.local_topology.west == -1
        s_new.u[start.x, y_slice] = -s.u[start.x + 1, y_slice]
        s_new.v[start.x, y_slice] =  s.v[start.x + 1, y_slice]
        s_new.h[start.x, y_slice] =  s.h[start.x + 1, y_slice]
    end

    if geometry.local_topology.east == -1
        s_new.u[stop.x, y_slice] = -s.u[stop.x - 1, y_slice]
        s_new.v[stop.x, y_slice] =  s.v[stop.x - 1, y_slice]
        s_new.h[stop.x, y_slice] =  s.h[stop.x - 1, y_slice]
    end

    return s_new
end

function shallow_water_dynamics(s_new::State{T}, s::State{T}, geometry::Geometry, comm, b::Matrix{T}, dt::T, dx::T, dy::T) where T <: AbstractFloat

    s_exc = exchange_state_halos(s, geometry, comm)
    s_new = apply_boundary_conditions(s_new, s_exc, geometry)
    s_new = apply_model(s_new, s_exc, geometry, b, dt, dx, dy)

    return s_new, s
end


function shallow_water_model(s::State{T}, geometry::Geometry, comm, b::Matrix{T}, n_steps::Int, dt::T, dx::T, dy::T) where T <: AbstractFloat

    s_new = create_null_state(geometry)

    for _ in 1:n_steps
        s, s_new = shallow_water_dynamics(s_new, s, geometry, comm, b, dt, dx, dy)
    end

    return s
end