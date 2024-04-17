

struct RectangularGrid
    nx::Int
    ny::Int
end


struct RectangularSubdomain
    partition_nx::Int
    partition_ny::Int
    start_x::Int
    start_y::Int
    local_nx::Int
    local_ny::Int
end

const CardinalTuple = NamedTuple{(:north,:south,:east,:west), Tuple{Int, Int, Int, Int}}


struct Geometry
    # Global PE Grid Info:
    nranks::Int       # Total number of MPI ranks (convience member, nranks == nxprocs * nyprocs)
    nxprocs::Int    # Size of the processor grid in the x direction
    nyprocs::Int    # Size of the processor grid in the y direction
    xproc::Int      # x-coord on the processor grid of this block (0-based)
    yproc::Int      # y-coord on the processor grid of this block (0-based)

    # Local PE Grid Info:
    local_rank::Int    # MPI ranks of this process
    local_topology::CardinalTuple   # Local process grid topology
    
    # Global Domain Info
    origin::SVector{2, Float64}      # The position of the global domain origin in real space
    extent::SVector{2, Float64}      # The extent of the global domain in real space
    grid_extent::SVector{2, Int} # The extent of the global domain in grid space
    
    # Local Domain Info
    local_grid_origin::SVector{2, Int} # The position of the local domain origin in grid space
    local_grid_extent::SVector{2, Int} # The extent of the local domain in grid space
    local_halo_depth::CardinalTuple  # The number of halo nodes in the N,S,E,and W directions
    local_ghost_depth::CardinalTuple # The number of ghost nodes in the N,S,E,and W directions
end


function get_locally_owned_range(geometry::Geometry)
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth
    start_x = 1 + halo_depth.west
    start_y = 1 + halo_depth.south
    end_x = (start_x + ghost_depth.west + geometry.local_grid_extent.x + ghost_depth.east) - 1
    end_y = (start_y + ghost_depth.south + geometry.local_grid_extent.y + ghost_depth.north) - 1

    return SVector{2, Int}(start_x, start_y), SVector{2, Int}(end_x, end_y)
end


function at_locally_owned(geometry::Geometry)
    start, stop = get_locally_owned_range(geometry)
    return start.x:stop.x, start.y:stop.y
end


function get_local_domain_range(geometry::Geometry)
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth
    start_x = 1 + ghost_depth.west + halo_depth.west
    start_y = 1 + ghost_depth.south + halo_depth.south
    end_x   = start_x + geometry.local_grid_extent.x - 1
    end_y   = start_y + geometry.local_grid_extent.y - 1

    return SVector{2, Int}(start_x, start_y), SVector{2, Int}(end_x, end_y)
end


function at_local_domain(geometry::Geometry)
    start, stop = get_local_domain_range(geometry)
    return start.x:stop.x, start.y:stop.y
end


function get_locally_active_range(geometry::Geometry)
    halo_depth = geometry.local_halo_depth
    ghost_depth = geometry.local_ghost_depth

    start_x = 1
    start_y = 1
    
    end_x = (halo_depth.west + ghost_depth.west + 
             geometry.local_grid_extent.x + 
             halo_depth.east + ghost_depth.east)
    
    end_y = (halo_depth.south + ghost_depth.south + 
             geometry.local_grid_extent.y + 
             halo_depth.north + ghost_depth.north)

    return SVector{2, Int}(start_x, start_y), SVector{2, Int}(end_x, end_y)
end


function get_locally_active_shape(geometry::Geometry)
    _, shape = get_locally_active_range(geometry)
    return (shape.x, shape.y)
end


function at_locally_active(geometry::Geometry)
    start, stop = get_locally_active_range(geometry)
    return start.x:stop.x, start.y:stop.y
end


function coord_to_index_xy_order(bounds::SVector{2, Int}, coord::SVector{2, Int})
    index = coord.x + bounds.x * (coord.y - 1)
    @assert 0 < index && index <= (bounds.x * bounds.y)
    return index
end


function index_to_coord_xy_order(bounds::SVector{2, Int}, index::Int)
    @assert 0 < index && index <= (bounds.x * bounds.y)
    y = div(index-1, bounds.x) + 1
    x = index - (bounds.x * (y-1))
    return SVector{2, Int}(x, y)
end


function split(x, n)
    @assert x >= n
 
    if (x % n == 0)
        return [div(x, n) for i in 1:n]
    else
        zp = n - (x % n)
        pp = div(x, n)
        return [i >= zp ? pp + 1 : pp for i in 0:(n-1)]
    end
end


function prefix_sum(arr)
    return accumulate(+, arr)
end


function partition_rectangular_grid(grid::RectangularGrid, num_subgrids)
    ratio = grid.nx / grid.ny

    if ratio > 1
        xy_ordered_pair = (smaller, larger) -> (larger, smaller)
    else
        xy_ordered_pair = (smaller, larger) -> (smaller, larger)
    end

    xy_factor_pairs = [xy_ordered_pair(i, div(num_subgrids, i)) for i in range(1, floor(Int, sqrt(num_subgrids))+1) if num_subgrids % i == 0]

    ratios = [a/b for (a,b) in xy_factor_pairs]

    idx = argmin(abs.(ratios .- ratio))

    nxprocs, nyprocs = xy_factor_pairs[idx]

    local_nx = split(grid.nx, nxprocs)
    start_x = prefix_sum([1; local_nx[1:end-1]])
    local_ny = split(grid.ny, nyprocs)
    start_y = prefix_sum([1; local_ny[1:end-1]])
    x_partition = collect(zip(start_x, local_nx))
    y_partition = collect(zip(start_y, local_ny))
    subgrid_extents = Iterators.product(x_partition, y_partition)
    create_subgrid = extents -> RectangularSubdomain(nxprocs, nyprocs, extents[1][1], extents[2][1], extents[1][2], extents[2][2])
    subgrid = vec(map(create_subgrid, subgrid_extents))
    return subgrid, nxprocs, nyprocs
end


function create_geometry(rank::Int, nranks::Int, grid::RectangularGrid, halo_depth::Int, ghost_depth::Int, global_origin::SVector{2, Float64}=SVector{2, Float64}(0.0,0.0), global_extent::SVector{2, Float64}=SVector{2, Float64}(1.0,1.0))
    subgrid_idx = rank + 1

    subgrids, nxprocs, nyprocs = partition_rectangular_grid(grid, nranks)
    @assert nranks == nxprocs * nyprocs

    local_subgrid = subgrids[subgrid_idx]

    bounds = SVector{2, Int}(nxprocs, nyprocs)
    coord = index_to_coord_xy_order(bounds, subgrid_idx)

    # We subtract 1 from the coord_to_index results to make them zero based rank indices.

    if coord.y != nyprocs
        north = coord_to_index_xy_order(bounds, SVector{2, Int}(coord.x, coord.y + 1)) - 1
    else
        north = -1
    end

    if coord.y != 1
        south = coord_to_index_xy_order(bounds, SVector{2, Int}(coord.x, coord.y - 1)) - 1
    else
        south = -1
    end

    if coord.x != nxprocs
        east = coord_to_index_xy_order(bounds, SVector{2, Int}(coord.x + 1, coord.y)) - 1
    else
        east = -1
    end

    if coord.x != 1
        west = coord_to_index_xy_order(bounds, SVector{2, Int}(coord.x - 1, coord.y)) - 1
    else
        west = -1
    end

    topology = CardinalTuple((north, south, east, west))
    halo = map(neighbor_id -> neighbor_id != -1 ? halo_depth : 0, topology)
    ghost = map(neighbor_id -> neighbor_id == -1 ? ghost_depth : 0, topology)
   
    geometry = Geometry(
        nranks,
        nxprocs,
        nyprocs,
        coord[1] - 1,
        coord[2] - 1,
        rank,
        topology,
        global_origin,
        global_extent,
        SVector{2, Int}(grid.nx, grid.ny),
        SVector{2, Int}(local_subgrid.start_x, local_subgrid.start_y),
        SVector{2, Int}(local_subgrid.local_nx, local_subgrid.local_ny),
        halo,
        ghost
        )


    return geometry
end