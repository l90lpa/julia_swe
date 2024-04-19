module julia_swe

    using StaticArrays
    using IterTools
    using MPI

    export Geometry, get_locally_owned_range, at_locally_owned,
        get_local_domain_range, at_local_domain, get_locally_active_range,
        get_locally_active_shape, at_locally_active, partition_rectangular_grid,
        create_geometry, index_to_coord_xy_order, coord_to_index_xy_order

    include("geometry.jl")

    export State

    include("state.jl")

    export create_tsunami_pulse_state, create_constant_height_state, create_null_state

    include("analytic_init.jl")

    export exchange_field_halos, exchange_state_halos

    include("exchange_halos.jl")

    export shallow_water_model

    include("model.jl")

    export gather_matrix

    include("utils.jl")

end # module julia_swe
