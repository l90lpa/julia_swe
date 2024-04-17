@testset "Geometry" begin
    @testset "Conversion between coord and index - 1" begin

        bounds = SVector{2, Int}(2, 2)

        proc_index = 1
        proc_coord = index_to_coord_xy_order(bounds, proc_index)
        @test proc_coord == SVector{2, Int}(1, 1)

        proc_coord = SVector{2, Int}(1, 1)
        proc_index = coord_to_index_xy_order(bounds, proc_coord)
        @test proc_index == 1

        proc_index = 2
        proc_coord = index_to_coord_xy_order(bounds, proc_index)
        @test proc_coord == SVector{2, Int}(2, 1)

        proc_coord = SVector{2, Int}(2, 1)
        proc_index = coord_to_index_xy_order(bounds, proc_coord)
        @test proc_index == 2

        proc_index = 3
        proc_coord = index_to_coord_xy_order(bounds, proc_index)
        @test proc_coord == SVector{2, Int}(1, 2)

        proc_coord = SVector{2, Int}(1, 2)
        proc_index = coord_to_index_xy_order(bounds, proc_coord)
        @test proc_index == 3
    end

    @testset "Conversion between coord and index - 2" begin

        bounds = SVector{2, Int}(3, 3)

        proc_index = 5
        proc_coord = index_to_coord_xy_order(bounds, proc_index)
        @test proc_coord == SVector{2, Int}(2, 2)

        proc_coord = SVector{2, Int}(2, 2)
        proc_index = coord_to_index_xy_order(bounds, proc_coord)
        @test proc_index == 5
    end

    @testset "Rectangular domain partitioning" begin
        domain = RectangularGrid(17,17)
        num_subdomains = 4

        subgrids, _, _ = partition_rectangular_grid(domain, num_subdomains)

        @test size(subgrids) == (4,)

        @test subgrids[1].start_x == 1 && subgrids[1].start_y == 1
        @test subgrids[1].local_nx == 8 && subgrids[1].local_ny == 8
        @test subgrids[2].start_x == 9 && subgrids[2].start_y == 1 
        @test subgrids[2].local_nx == 9 && subgrids[2].local_ny == 8
        @test subgrids[3].start_x == 1 && subgrids[3].start_y == 9
        @test subgrids[3].local_nx == 8 && subgrids[3].local_ny == 9
        @test subgrids[4].start_x == 9 && subgrids[4].start_y == 9
        @test subgrids[4].local_nx == 9 && subgrids[4].local_ny == 9

        @test subgrids[4].start_x == (subgrids[1].local_nx + 1) && subgrids[4].start_y == (subgrids[1].local_ny + 1)
    end

    @testset "Geometry creation - 1" begin
        
        rank = 3
        nranks = 4
        grid = RectangularGrid(17,17)
        halo_depth = 1
        ghost_depth = 1

        geometry = create_geometry(rank, nranks, grid, halo_depth, ghost_depth)

        @test geometry.nxprocs == 2 && geometry.nyprocs == 2
        @test geometry.xproc == 1 && geometry.yproc == 1

        @test geometry.local_grid_origin.x == 9 && geometry.local_grid_origin.y == 9
        @test geometry.local_grid_extent.x == 9 && geometry.local_grid_extent.y == 9

        @test geometry.local_halo_depth.north == 0
        @test geometry.local_halo_depth.south == 1
        @test geometry.local_halo_depth.east == 0
        @test geometry.local_halo_depth.west == 1

        @test geometry.local_ghost_depth.north == 1
        @test geometry.local_ghost_depth.south == 0
        @test geometry.local_ghost_depth.east == 1
        @test geometry.local_ghost_depth.west == 0

        @test geometry.local_topology.north == -1
        @test geometry.local_topology.south == 1
        @test geometry.local_topology.east == -1
        @test geometry.local_topology.west == 2
    end

    @testset "Geometry creation - 2" begin
        
        rank = 4
        nranks = 9
        grid = RectangularGrid(3,3)
        halo_depth = 1
        ghost_depth = 1

        geometry = create_geometry(rank, nranks, grid, halo_depth, ghost_depth)

        @test geometry.nxprocs == 3 && geometry.nyprocs == 3
        @test geometry.xproc == 1 && geometry.yproc == 1

        @test geometry.local_grid_origin.x == 2 && geometry.local_grid_origin.y == 2
        @test geometry.local_grid_extent.x == 1 && geometry.local_grid_extent.y == 1

        @test geometry.local_halo_depth.north == 1
        @test geometry.local_halo_depth.south == 1
        @test geometry.local_halo_depth.east == 1
        @test geometry.local_halo_depth.west == 1

        @test geometry.local_ghost_depth.north == 0
        @test geometry.local_ghost_depth.south == 0
        @test geometry.local_ghost_depth.east == 0
        @test geometry.local_ghost_depth.west == 0

        @test geometry.local_topology.north == 7
        @test geometry.local_topology.south == 1
        @test geometry.local_topology.east == 5
        @test geometry.local_topology.west == 3
    end

    @testset "Geometry indexing - 1" begin
        
        rank = 3
        nranks = 4
        grid = RectangularGrid(6,6)
        halo_depth = 1
        ghost_depth = 2

        geometry = create_geometry(rank, nranks, grid, halo_depth, ghost_depth)

        @test get_locally_active_shape(geometry) == (6,6)
        @test at_locally_active(geometry) == (1:6,1:6)
        @test at_locally_owned(geometry) == (2:6,2:6)
        @test at_local_domain(geometry) == (2:4,2:4)
    end

    @testset "Geometry indexing - 2" begin
        
        rank = 1
        nranks = 2
        grid = RectangularGrid(2,1)
        halo_depth = 1
        ghost_depth = 0

        geometry = create_geometry(rank, nranks, grid, halo_depth, ghost_depth)

        @test get_locally_active_shape(geometry) == (2,1)
        @test at_locally_active(geometry) == (1:2,1:1)
        @test at_locally_owned(geometry) == (2:2,1:1)
        @test at_local_domain(geometry) == (2:2,1:1)
    end
end