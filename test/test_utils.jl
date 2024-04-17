using julia_swe
using MPI
using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
root = 0

@testset "Gather block matrix" begin
        
    if nranks == 1

        block = [1 2; 3 4]
        block_coord = (1,1)
        block_layout = (1,1)
        matrix = gather_matrix(block, block_coord, block_layout, comm, root)
        @test matrix == block
    elseif nranks == 2
        
        block_layout = (2,1)
        if rank == 0
            block = [1 2; 3 4]
            block_coord = (1,1)
            matrix = gather_matrix(block, block_coord, block_layout, comm, root)
            @test matrix == [1 2; 3 4; 5 6; 7 8]
        else
            block = [5 6; 7 8]
            block_coord = (2,1)
            matrix = gather_matrix(block, block_coord, block_layout, comm, root)
        end

    elseif nranks == 4
        
        block_layout = (2,2)
        if rank == 0
            block = [1 1; 1 1]
            block_coord = (1,1)
            matrix = gather_matrix(block, block_coord, block_layout, comm, root)
            @test matrix == [1 1 2 2 2; 1 1 2 2 2; 3 3 4 4 4; 3 3 4 4 4; 3 3 4 4 4; 3 3 4 4 4]
        elseif rank == 1
            block = [2 2 2; 2 2 2]
            block_coord = (1,2)
            gather_matrix(block, block_coord, block_layout, comm, root)
        elseif rank == 2
            block = [3 3; 3 3; 3 3; 3 3]
            block_coord = (2,1)
            gather_matrix(block, block_coord, block_layout, comm, root)
        else
            block = [4 4 4; 4 4 4; 4 4 4; 4 4 4]
            block_coord = (2,2)
            gather_matrix(block, block_coord, block_layout, comm, root)
        end

    else

        block_layout = (1,nranks)
        block = [rank rank; rank rank]
        block_coord = (1,rank+1)
        matrix = gather_matrix(block, block_coord, block_layout, comm, root)
        
        if rank == root
            test_matrix = Matrix{Int}(undef, 2, 2*nranks)
            for i in 1:nranks
                test_matrix[1, 2*i-1] = i-1
                test_matrix[1,   2*i] = i-1
                test_matrix[2, 2*i-1] = i-1
                test_matrix[2,   2*i] = i-1
            end
        
            @test matrix == test_matrix
        end
    end
end

MPI.Finalize()