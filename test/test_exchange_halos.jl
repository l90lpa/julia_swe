using julia_swe
using MPI
using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
root = 0


@testset "Exchange field halos - 1" begin
    @assert nranks <= 4

    if nranks == 1
        grid = SVector{2, Int}(1,1)
    elseif nranks == 2
        grid = SVector{2, Int}(2,1)
    elseif nranks == 4
        grid = SVector{2, Int}(2,2)
    else
        grid = SVector{2, Int}(1,nranks)
    end

    halo_depth = 1
    ghost_depth = 0
    geometry = create_geometry(rank, nranks, grid, halo_depth, ghost_depth)
    
    field = zeros(get_locally_active_shape(geometry))
    field[at_locally_owned(geometry)...] .= rank + 1
    new_field = exchange_field_halos(field, geometry, comm)

    if nranks == 1
        @test new_field == [1.0;;]
    elseif nranks == 2
        if rank == 0
            @test new_field == [1.0; 2.0;;]
        else
            @test new_field == [1.0; 2.0;;]
        end
    elseif nranks == 4
        if rank == 0
            @test new_field == [1.0 3.0; 2.0 0.0]
        elseif rank == 1
            @test new_field == [1.0 0.0; 2.0 4.0]
        elseif rank == 2
            @test new_field == [1.0 3.0; 0.0 4.0]
        else
            @test new_field == [0.0 3.0; 2.0 4.0]
        end
    else
        if rank == 0
            @test new_field == Matrix{Float64}([rank+1 rank+2])
        elseif rank == (nranks - 1)
            @test new_field == Matrix{Float64}([rank rank+1])
        else
            @test new_field == Matrix{Float64}([rank rank+1 rank+2])
        end
    end
end

MPI.Finalize()