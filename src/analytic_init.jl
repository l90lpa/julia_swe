
function create_local_field_tsunami_height(geometry::Geometry)
    # The global domain and grid must be square
    @assert geometry.extent.x == geometry.extent.y
    @assert geometry.grid_extent.x == geometry.grid_extent.y

    ymax = xmax = geometry.extent.x
    ny   = nx   = geometry.grid_extent.x
    dy   = dx   = xmax / (nx - 1)

    h = zeros(get_locally_active_shape(geometry))
    xmid = (xmax / 2.0) + geometry.origin.x
    ymid = (ymax / 2.0) + geometry.origin.y
    sigma = floor((xmax + 2 * dx) / 20.0)

    # Create a height field with a tsunami pulse
    local_origin_x = geometry.local_grid_origin.x
    local_origin_y = geometry.local_grid_origin.y
    x_slice, y_slice = at_locally_owned(geometry)
  
    for j in y_slice
        for i in x_slice
            dsqr = (((i-1) + (local_origin_x-1)) * dx - xmid) ^ 2 + (((j-1) + (local_origin_y-1)) * dy - ymid) ^ 2
            h[i, j] = 5000.0 + 30.0 * exp(-dsqr / sigma ^ 2)
        end
    end

    return h
end


function create_tsunami_pulse_state(geometry::Geometry)
    u = zeros(get_locally_active_shape(geometry))
    v = zeros(get_locally_active_shape(geometry))
    h = create_local_field_tsunami_height(geometry)
    return State(u, v, h)
end


function create_constant_height_state(geometry::Geometry, h0::Float64)
    u = zeros(get_locally_active_shape(geometry))
    v = zeros(get_locally_active_shape(geometry))
    h = h0 .* ones(get_locally_active_shape(geometry))
    return State(u, v, h)
end


function create_null_state(geometry::Geometry)
    u = zeros(get_locally_active_shape(geometry))
    v = zeros(get_locally_active_shape(geometry))
    h = zeros(get_locally_active_shape(geometry))
    return State(u, v, h)
end