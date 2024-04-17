function gather_matrix(block::Matrix{T}, block_coord::Tuple{Int, Int}, block_layout::Tuple{Int, Int}, comm, root::Int) where T<:Number
    # Block coords are 1-based

    if block_layout[1] * block_layout[2] == 1
        return block
    end
    
    rank = MPI.Comm_rank(comm)

    num_blocks = block_layout[1] * block_layout[2]
    
    # gather block coords
    size_and_location = Array{Int, 1}(undef, 4)
    size_and_location[1] = size(block, 1) # number of rows in block
    size_and_location[2] = size(block, 2) # number of columns in block
    size_and_location[3] = block_coord[1] # block row
    size_and_location[4] = block_coord[2] # block column
    global_size_and_location = MPI.Gather(size_and_location, comm, root=root)
    if rank == root
        global_size_and_location = reshape(global_size_and_location, 4, MPI.Comm_size(comm))
    end


    block_list = nothing
    block_list_buf = nothing
    nrows = nothing
    ncols = nothing
    if rank == root
        # construct buffer for global block list

        nrows = Vector{Int}(undef, block_layout[1])
        ncols = Vector{Int}(undef, block_layout[2])
        
        for i in 1:num_blocks
            if global_size_and_location[4, i] == 1
                nrows[global_size_and_location[3, i]] = global_size_and_location[1, i]
            end
            if global_size_and_location[3, i] == 1
                ncols[global_size_and_location[4, i]] = global_size_and_location[2, i]
            end
        end
        
        nrows = [0; accumulate(+, nrows)]
        ncols = [0; accumulate(+, ncols)]
        @assert length(nrows)-1 == block_layout[1] && length(ncols)-1 == block_layout[2]

        block_counts = Array{Int, 1}(undef, num_blocks)
        for i in 1:num_blocks
            block_counts[i] = global_size_and_location[1,i] * global_size_and_location[2,i]
        end
        
        block_list = Array{T, 1}(undef, nrows[end] * ncols[end])
        block_list_buf = MPI.VBuffer(block_list, block_counts)
    end

    # gather blocks
    MPI.Gatherv!(block, block_list_buf, comm)
    
    if rank == root
        # construct global matrix

        matrix = Matrix{T}(undef, nrows[end], ncols[end])
        
        for i in 1:num_blocks
            start_row = nrows[global_size_and_location[3, i]] + 1
            stop_row = nrows[global_size_and_location[3, i]] + global_size_and_location[1, i]
            start_col = ncols[global_size_and_location[4, i]] + 1
            stop_col = ncols[global_size_and_location[4, i]] + global_size_and_location[2, i]
            matrix[start_row:stop_row, start_col:stop_col] = block_list[block_list_buf.displs[i]+1 : block_list_buf.displs[i]+block_list_buf.counts[i]]
        end
        return matrix
    else
        return nothing
    end
end