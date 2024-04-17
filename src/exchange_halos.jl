
function exchange_field_halos(field::Matrix{T}, geometry::Geometry, comm) where T <: AbstractFloat

    if (geometry.nxprocs * geometry.nyprocs) == 1
        return copy(field)
    end

    local_topology = geometry.local_topology
    halo_depth = geometry.local_halo_depth

    start, stop = get_locally_owned_range(geometry)

    # send buffer slices
    halo_source_slices = Dict([
        ("north", (start.x : stop.x, stop.y - halo_depth.north + 1 : stop.y)), 
        ("south", (start.x : stop.x, start.y : start.y + halo_depth.south - 1)),    
        ("east" , (stop.x - halo_depth.east + 1 : stop.x, start.y : stop.y)),   
        ("west" , (start.x : start.x + halo_depth.west - 1, start.y : stop.y))
    ])
    
    # recv buffer slices
    halo_slices = Dict([
        ("north", (start.x : stop.x, stop.y + 1 : stop.y + halo_depth.north)),
        ("south", (start.x : stop.x, start.y - halo_depth.south : start.y - 1)),      
        ("east" , (stop.x + 1 : stop.x + halo_depth.east, start.y : stop.y)),  
        ("west" , (start.x - halo_depth.west : start.x - 1, start.y : stop.y))
    ])
    
    neighbor_ids = Dict([
        ("north", local_topology.north),
        ("south", local_topology.south),
        ("east" ,  local_topology.east),
        ("west" ,  local_topology.west)
    ])
    
    send_recv_pairs = [
        ("south", "north"),
        ("west", "east"),
        ("north", "south"),
        ("east", "west"),
    ]

    new_field = copy(field)

    for (send_name, recv_name) in send_recv_pairs
        send_id = neighbor_ids[send_name]
        recv_id = neighbor_ids[recv_name]
        
        if send_id == -1 && recv_id == -1
            continue
        elseif send_id == -1
            recv_buf = field[halo_slices[recv_name]...]
            MPI.Recv!(recv_buf, comm, source=recv_id)
            new_field[halo_slices[recv_name]...] = recv_buf
        elseif recv_id == -1
            send_buf = field[halo_source_slices[send_name]...]
            MPI.Send(send_buf, comm, dest=send_id)
        else
            recv_buf = field[halo_slices[recv_name]...]
            send_buf = field[halo_source_slices[send_name]...]
            MPI.Sendrecv!(send_buf, recv_buf, comm, source=recv_id, dest=send_id)
            new_field[halo_slices[recv_name]...] = recv_buf
        end
    end

    return new_field
end


function exchange_state_halos(s::State{T}, geometry::Geometry, comm) where T <: AbstractFloat

    u_new = exchange_field_halos(s.u, geometry, comm)
    v_new = exchange_field_halos(s.v, geometry, comm)
    h_new = exchange_field_halos(s.h, geometry, comm)

    return State{T}(u_new, v_new, h_new)
end