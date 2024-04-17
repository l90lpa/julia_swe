using julia_swe
using MPI
using Test
using StaticArrays

include("test_geometry.jl")

@testset "Utils" begin
    run(`$(mpiexec()) -n 1 $(Base.julia_cmd()) ./test_utils.jl`)
    run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) ./test_utils.jl`)
    run(`$(mpiexec()) -n 3 $(Base.julia_cmd()) ./test_utils.jl`)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) ./test_utils.jl`)
end

@testset "Exchange Halos" begin
    run(`$(mpiexec()) -n 1 $(Base.julia_cmd()) ./test_exchange_halos.jl`)
    run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) ./test_exchange_halos.jl`)
    run(`$(mpiexec()) -n 3 $(Base.julia_cmd()) ./test_exchange_halos.jl`)
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) ./test_exchange_halos.jl`)
end

@testset "Model" begin
    run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) ./test_model.jl`)
end