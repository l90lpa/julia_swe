using julia_swe
using MPI
using Test
using StaticArrays

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
root = 0


function fixture_create_geometry(rank::Int, nranks::Int, grid::RectangularGrid, extent::SVector{2, Float64})
    dx = extent.x / (grid.nx - 1)
    dy = extent.x / (grid.nx - 1)
    adjusted_extent = SVector{2, Float64}(extent.x - 2 * dx, extent.y - 2 * dy)
    adjusted_grid = RectangularGrid(grid.nx - 2, grid.ny - 2)
    geometry = create_geometry(rank, nranks, adjusted_grid, 1, 1, SVector{2, Float64}(dx,dy), adjusted_extent)
    return geometry
end


@testset "Null IC" begin

    xmax = ymax = 100000.0
    nx = ny = 101
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 1

    grid = RectangularGrid(nx, ny)
    geometry = fixture_create_geometry(rank, nranks, grid, SVector{2, Float64}(xmax, ymax))

    s = create_null_state(geometry)

    b = zeros(get_locally_active_shape(geometry))

    s_new = shallow_water_model(s, geometry, comm, b, num_steps, dt, dx, dy)

    u_local = s_new.u[at_locally_owned(geometry)...]
    v_local = s_new.v[at_locally_owned(geometry)...]
    h_local = s_new.h[at_locally_owned(geometry)...]
    
    @test all(u_local .== 0.0)
    @test all(v_local .== 0.0)
    @test all(h_local .== 0.0)
end


@testset "Constant height IC" begin

    xmax = ymax = 100000.0
    nx = ny = 101
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 1

    grid = RectangularGrid(nx, ny)
    geometry = fixture_create_geometry(rank, nranks, grid, SVector{2, Float64}(xmax, ymax))
    
    s = create_constant_height_state(geometry, 10.0)

    b = zeros(get_locally_active_shape(geometry))

    s_new = shallow_water_model(s, geometry, comm, b, num_steps, dt, dx, dy)

    u_local = s_new.u[at_locally_owned(geometry)...]
    v_local = s_new.v[at_locally_owned(geometry)...]
    h_local = s_new.h[at_locally_owned(geometry)...]

    @test all(u_local .== 0.0)
    @test all(v_local .== 0.0)
    @test all(h_local .== 10.0)
end


@testset "Tsunami pulse IC" begin

    xmax = ymax = 10000.0
    nx = ny = 11
    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030.0)
    num_steps = 100

    grid = RectangularGrid(nx, ny)
    geometry = fixture_create_geometry(rank, nranks, grid, SVector{2, Float64}(xmax, ymax))

    s = create_tsunami_pulse_state(geometry)

    b = zeros(get_locally_active_shape(geometry))

    s_new = shallow_water_model(s, geometry, comm, b, num_steps, dt, dx, dy)

    u_global = gather_matrix(s_new.u[at_locally_owned(geometry)...], (geometry.xproc + 1, geometry.yproc + 1), (geometry.nxprocs, geometry.nyprocs), comm, root)
    v_global = gather_matrix(s_new.v[at_locally_owned(geometry)...], (geometry.xproc + 1, geometry.yproc + 1), (geometry.nxprocs, geometry.nyprocs), comm, root)
    h_global = gather_matrix(s_new.h[at_locally_owned(geometry)...], (geometry.xproc + 1, geometry.yproc + 1), (geometry.nxprocs, geometry.nyprocs), comm, root)

    if rank == root
        rms_u = sqrt(sum(u_global .^ 2) / (nx * ny))
        rms_v = sqrt(sum(v_global .^ 2) / (nx * ny))
        rms_h = sqrt(sum(h_global .^ 2) / (nx * ny))

        # println("rms_u = ", rms_u, ", rms_v = ", rms_v, ", rms_h = ", rms_h)
        @test abs(rms_u - 0.0016110910527818616) < 1.0e-12
        @test abs(rms_v - 0.0016110910527825017) < 1.0e-12
        @test abs(rms_h - 5000.371968791809)     < 1.0e-12
    end
end

MPI.Finalize()