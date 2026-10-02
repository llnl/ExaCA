// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_CELLDATA_HPP
#define EXACA_CELLDATA_HPP

#include "CAconfig.hpp"
#include "CAgrid.hpp"
#include "CAinputdata.hpp"
#include "CAparsefiles.hpp"
#include "CAtemperature.hpp"
#include "CAtypes.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Data that belongs to individual cells governing the current state and various ID values
template <typename MemorySpace>
struct CellData {

    using memory_space = MemorySpace;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_int_host = typename view_type_int::host_mirror_type;
    using device_layout = typename view_type_int::array_layout;
    using view_type_int_2d_host = Kokkos::View<int **, device_layout, Kokkos::HostSpace>;
    using view_type_int_unmanaged = Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using view_type_bool = Kokkos::View<bool *, memory_space>;
    using view_type_short = Kokkos::View<short *, memory_space>;
    using view_type_float = Kokkos::View<float *, memory_space>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    int next_layer_first_epitaxial_grain_id = 1;
    int num_prior_nuclei = 0;
    view_type_int grain_id_all_layers, cell_type;
    view_type_short phase_id_all_layers;
    view_type_bool melt_edge, melt_edge_all_layers;
    // Substrate inputs from file
    SubstrateInputs _inputs;
    // Storing of whether or not a cell is at an edge of a melt pool (FromFile and FromFinch problem types only)
    bool _store_melt_pool_edge;

    // Constructor for views and view bounds for current layer
    // GrainID/PhaseID are initialized to zeros, while others are not initialized
    // cell_type only exists for the current layer of a multilayer problem
    CellData(const Grid &grid, SubstrateInputs inputs, const bool store_melt_pool_edge = false)
        : grain_id_all_layers(view_type_int("GrainID", grid.domain_size_all_layers))
        , cell_type(view_type_int(Kokkos::ViewAllocateWithoutInitializing("cell_type"), grid.domain_size))
        , phase_id_all_layers(view_type_short("PhaseID", grid.domain_size_all_layers))
        , _inputs(inputs)
        , _store_melt_pool_edge(store_melt_pool_edge) {
        if (_store_melt_pool_edge) {
            // Default init to zero
            melt_edge_all_layers = view_type_bool("melt_edge", grid.domain_size_all_layers);
            // Current layer
            getCurrentLayerMeltEdge(grid.layer_range);
        }
    }

    // Get the subview associated with the edge indicator of cells in the current layer
    void getCurrentLayerMeltEdge(std::pair<int, int> layer_range) {
        melt_edge = Kokkos::subview(melt_edge_all_layers, layer_range);
    }

    // Initializes the single active cell and associated active cell data structures for the single grain at the domain
    // center
    void initSubstrate_SingleGrain(const int id, const Grid &grid) {

        // Location of the single grain
        int grain_location_x = Kokkos::floorf(static_cast<float>(grid.nx) / 2.0);
        int grain_location_y = Kokkos::floorf(static_cast<float>(grid.ny) / 2.0);
        int grain_location_z = Kokkos::floorf(static_cast<float>(grid.nz) / 2.0);

        // Local copies for lambda capture.
        auto cell_type_local = cell_type;
        auto grain_id_all_layers_local = grain_id_all_layers;
        int single_grain_orientation_local = _inputs.single_grain_orientation;

        auto policy = Kokkos::RangePolicy<execution_space>(0, grid.domain_size);
        Kokkos::parallel_for(
            "SingleGrainInit", policy, KOKKOS_LAMBDA(const int &index) {
                int coord_x = grid.getCoordX(index);
                int coord_y = grid.getCoordY(index);
                int coord_z = grid.getCoordZ(index);
                int coord_y_global = coord_y + grid.y_offset;
                if ((coord_x == grain_location_x) && (coord_y_global == grain_location_y) &&
                    (coord_z == grain_location_z)) {
                    cell_type_local(index) = Active;
                    grain_id_all_layers_local(index) = single_grain_orientation_local + 1;
                }
                else
                    cell_type_local(index) = Liquid;
            });
        if ((grain_location_y >= grid.y_offset) && (grain_location_y < grid.y_offset + grid.ny_local))
            std::cout << "Rank " << id << " initialized a grain with orientation " << single_grain_orientation_local
                      << " initialized at X = " << grain_location_x << ", Y = " << grain_location_y
                      << ", Z = " << grain_location_z << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Grain struct initialized" << std::endl;
    }

    // Get the X, Y coordinates and grain ID values for grains at the bottom surface for problem type Directional
    view_type_int_2d_host getSurfaceActiveCellData(int &substrate_act_cells, const Grid &grid,
                                                   const unsigned long rng_seed, const int n_grain_orientations,
                                                   const bool is_periodic) {

        // Number of cells at the bottom surface that could potentially be assigned GrainID values
        int bottom_surface_size;
        if (is_periodic)
            bottom_surface_size = (grid.nx - 2) * (grid.ny - 2);
        else
            bottom_surface_size = grid.nx * grid.ny;

        // First get number of substrate grains for each initialization condition
        if (_inputs.surface_init_mode == "SurfaceSiteFraction")
            substrate_act_cells = Kokkos::round(_inputs.fract_surface_sites_active * bottom_surface_size);
        else if (_inputs.surface_init_mode == "SurfaceSiteDensity")
            substrate_act_cells =
                Kokkos::round(_inputs.surface_site_density * static_cast<double>(bottom_surface_size) * grid.deltax *
                              grid.deltax * pow(10, 12));
        if (_inputs.surface_init_mode == "Custom")
            substrate_act_cells = _inputs.grain_locations_x.size();

        // View for storing surface grain locations and IDs
        view_type_int_2d_host act_cell_data_host(Kokkos::ViewAllocateWithoutInitializing("ActCellData_Host"),
                                                 substrate_act_cells, 3);
        if (_inputs.surface_init_mode == "SurfaceSiteFraction") {
            // Fraction of surface sites active was given - ensure the appropriate number of cells are assigned
            // GrainIDs. Note that the physical locations of these sites (x,y) will vary based on the domain size/cell
            // size Create list of grain IDs and shuffle - leave 0s for cells without substrate grains
            std::vector<int> grain_locations_1d(bottom_surface_size, 0);
            if (_inputs.shuffle_grain_orientations) {
                // Assign substrate_act_cells grain ID values, randomly selected from list of n_grain_orientations
                std::vector<int> shuffled_orientations_list(n_grain_orientations);
                // shuffled_orientations_list contains orientations 0 through n_grain_orientations-1
                for (int i = 0; i < n_grain_orientations; i++)
                    shuffled_orientations_list[i] = i;
                std::mt19937_64 ori_gen(rng_seed);
                std::shuffle(shuffled_orientations_list.begin(), shuffled_orientations_list.end(), ori_gen);
                // Assign randomly permuted grain orientations, grain IDs 1 through substrate_act_cells
                for (int n = 0; n < substrate_act_cells; n++) {
                    int n_repeat = n / n_grain_orientations;
                    int n_unique = n % n_grain_orientations;
                    grain_locations_1d[n] = (n_repeat + 1) * shuffled_orientations_list[n_unique] + 1;
                }
            }
            else {
                // Assign grain IDs 1 through substrate_act_cells
                for (int n = 0; n < substrate_act_cells; n++) {
                    grain_locations_1d[n] = n + 1;
                }
            }
            std::mt19937_64 gen(rng_seed);
            std::shuffle(grain_locations_1d.begin(), grain_locations_1d.end(), gen);
            // Fill act_cell_data_host with the non-zero grain IDs and their associated X and Y based on the position in
            // the 1D vector
            int act_cell_count = 0;
            for (int n = 0; n < bottom_surface_size; n++) {
                if (grain_locations_1d[n] != 0) {
                    // Periodic BCs: only place grains between X,Y = 1 through nx-1,ny-1
                    act_cell_data_host(act_cell_count, 0) = grid.getCoordXGlobal(n, is_periodic);
                    act_cell_data_host(act_cell_count, 1) = grid.getCoordYGlobal(n, is_periodic);
                    act_cell_data_host(act_cell_count, 2) = grain_locations_1d[n];
                    act_cell_count++;
                }
            }
        }
        else if (_inputs.surface_init_mode == "SurfaceSiteDensity") {
            // Calls to Xdist(gen) and Y dist(gen) return random locations for grain seeds
            // Since X = 0 and X = nx-1 are the cell centers of the last cells in X, locations are evenly scattered
            // between X = -0.49999 and X = nx - 0.5, as the cells have a half width of 0.5. Note that if the number of
            // grains is large compared to the number of cells, multiple grain IDs may be assigned to one cell and the
            // total density will be underestimated on the given grid. Do not place grains around edges if periodic BCs
            std::mt19937_64 gen(rng_seed);
            std::uniform_real_distribution<double> x_dist(-0.49999 + is_periodic, grid.nx - 0.5 - is_periodic);
            std::uniform_real_distribution<double> y_dist(-0.49999 + is_periodic, grid.ny - 0.5 - is_periodic);
            // Randomly locate substrate grain seeds for cells in the interior of this subdomain (at the k = 0 bottom
            // surface)
            for (int n = 0; n < substrate_act_cells; n++) {
                double x_location = x_dist(gen);
                double y_location = y_dist(gen);
                // Randomly select integer coordinates between 0 and nx-1 or ny-1
                act_cell_data_host(n, 0) = Kokkos::round(x_location);
                act_cell_data_host(n, 1) = Kokkos::round(y_location);
                act_cell_data_host(n, 2) = n + 1; // grain ID for epitaxial seeds must be > 0
            }
        }
        else if (_inputs.surface_init_mode == "Custom") {
            // Values were already given in the inputs struct, copy these to the view
            for (int n = 0; n < substrate_act_cells; n++) {
                act_cell_data_host(n, 0) = _inputs.grain_locations_x[n];
                act_cell_data_host(n, 1) = _inputs.grain_locations_y[n];
                act_cell_data_host(n, 2) = _inputs.grain_ids[n];
            }
        }
        return act_cell_data_host;
    }

    // Initializes cell types and epitaxial Grain ID values where substrate grains are future active cells on the bottom
    // surface of the constrained domain
    void initSubstrate_Directional(const int id, const int np, const Grid &grid, const unsigned long rng_seed,
                                   const int n_grain_orientations = 0, const bool is_periodic = false) {

        // Fill the view of cell X, Y, and ID values, updating the number of substrate active cells appropriately
        // TODO: Could generate random numbers on GPU, instead of using host view and copying over - but would also need
        // inputs struct to store device data for grain locations in X, Y, and GrainIDs
        int substrate_act_cells;
        view_type_int_2d_host act_cell_data_host =
            getSurfaceActiveCellData(substrate_act_cells, grid, rng_seed, n_grain_orientations, is_periodic);
        // Copy views of substrate grain locations and IDs back to the device
        auto act_cell_data = Kokkos::create_mirror_view_and_copy(memory_space(), act_cell_data_host);

        // Start with all cells as liquid prior to locating substrate grain seeds
        Kokkos::deep_copy(cell_type, Liquid);

        // Local copies for lambda capture.
        auto cell_type_local = cell_type;
        auto grain_id_all_layers_local = grain_id_all_layers;

        // Determine which grains/active cells belong to which MPI ranks
        auto policy = Kokkos::RangePolicy<execution_space>(0, substrate_act_cells);
        Kokkos::parallel_for(
            "ConstrainedGrainInit", policy, KOKKOS_LAMBDA(const int &n) {
                // What are the X and Y coordinates of this active cell relative to the X and Y bounds of this rank?
                if ((act_cell_data(n, 1) >= grid.y_offset) && (act_cell_data(n, 1) < grid.y_offset + grid.ny_local)) {
                    // Convert X and Y coordinates to values relative to this MPI rank's grid (Z = 0 for these active
                    // cells, at bottom surface) GrainIDs come from the position on the list of substrate active cells
                    // to avoid reusing the same value
                    int coord_x = act_cell_data(n, 0);
                    int coord_y = act_cell_data(n, 1) - grid.y_offset;
                    int coord_z = 0;
                    int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                    cell_type_local(index) = Active;
                    grain_id_all_layers_local(index) = act_cell_data(n, 2); // assign GrainID > 0 to epitaxial seeds
                }
            });
        // Option to fill empty sites at bottom surface with the grain ID of the nearest grain
        if (_inputs.fill_bottom_surface) {
            int lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y;
            if (is_periodic) {
                lower_bound_x = 1;
                upper_bound_x = grid.nx - 1;
                if (id == 0)
                    lower_bound_y = 1;
                else
                    lower_bound_y = 0;
                if (id == np - 1)
                    upper_bound_y = grid.ny_local - 1;
                else
                    upper_bound_y = grid.ny_local;
            }
            else {
                lower_bound_x = 0;
                upper_bound_x = grid.nx;
                lower_bound_y = 0;
                upper_bound_y = grid.ny_local;
            }
            auto md_policy =
                Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                    {lower_bound_x, lower_bound_y}, {upper_bound_x, upper_bound_y});
            // For cells that are not associated with grain centers, optionally assign them the GrainID of the nearest
            // grain center
            Kokkos::parallel_for(
                "BaseplateGen", md_policy, KOKKOS_LAMBDA(const int coord_x, const int coord_y) {
                    int index_all_layers = grid.get1DIndex(coord_x, coord_y, 0);
                    if (grain_id_all_layers_local(index_all_layers) == 0) {
                        // This cell needs to be assigned a GrainID value
                        // Check each possible baseplate grain center to find the closest one
                        float min_distance_to_this_grain = grid.nx * grid.ny;
                        int min_distance_to_this_grain_grain_id = 0;
                        for (int n = 0; n < substrate_act_cells; n++) {
                            // Substrate grain center at coord_x_grain, coord_y_grain - how far is the cell at i,
                            // j+y_offset?
                            int coord_y_grain_global = act_cell_data(n, 1);
                            int coord_x_grain = act_cell_data(n, 0);
                            int coord_y_global = coord_y + grid.y_offset;
                            float distance_to_this_grain_x = Kokkos::abs(coord_x - coord_x_grain);
                            float distance_to_this_grain_y = Kokkos::abs(coord_y_global - coord_y_grain_global);
                            // Optionally consider periodic boundary condition in distance calculation
                            if (is_periodic) {
                                distance_to_this_grain_x =
                                    Kokkos::fmin(distance_to_this_grain_x, grid.nx - distance_to_this_grain_x);
                                distance_to_this_grain_y =
                                    Kokkos::fmin(distance_to_this_grain_y, grid.ny - distance_to_this_grain_y);
                            }
                            float distance_to_this_grain =
                                Kokkos::sqrt(distance_to_this_grain_x * distance_to_this_grain_x +
                                             distance_to_this_grain_y * distance_to_this_grain_y);
                            if (distance_to_this_grain < min_distance_to_this_grain) {
                                // This is the closest grain center to cell at "CAGridLocation" - update values
                                min_distance_to_this_grain = distance_to_this_grain;
                                min_distance_to_this_grain_grain_id = act_cell_data(n, 2);
                            }
                        }
                        // GrainID associated with the closest baseplate grain center
                        grain_id_all_layers_local(index_all_layers) = min_distance_to_this_grain_grain_id;
                        // These cells are also future active cells
                        // For directional solidification, only one "layer" so index for all layers can be used here to
                        // index cell_type
                        cell_type_local(index_all_layers) = Active;
                    }
                });
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Number of substrate active cells across all ranks: " << substrate_act_cells << std::endl;
            std::cout << "Grain struct initialized" << std::endl;
        }
    }

    // Initializes substrate grain structure using either a Voronoi assignment of grain ID values to cells or reading
    // the data from a file
    void initSubstrate_BaseplatePowder(const int id, const Grid &grid, const unsigned long rng_seed) {

        // Determine the number of cells in the Z direction that are part of the baseplate
        int baseplate_size_z = getBaseplateSizeZ(id, grid);
        // Generate the baseplate microstructure, or read it from a file, to initialize the grain ID values from Z = 0
        // up to but not including Z = BaseplateTopZ
        if (_inputs.use_substrate_file)
            initBaseplateGrainID(id, grid, baseplate_size_z);
        else
            initVoronoiGrainID(id, grid, 0, baseplate_size_z, _inputs.baseplate_grain_spacing, rng_seed, "baseplate");

        // Powder layer extends from Z = powder_bottom_z up to but not including Z = powder_top_z
        // Bottom of layer is the next coordinate up from the baseplate
        int powder_bottom_z = Kokkos::round((_inputs.baseplate_top_z - grid.z_min) / grid.deltax) + 1;
        int powder_top_z = Kokkos::round((grid.z_max_layer[0] - grid.z_min) / grid.deltax) + 1;
        // Generate powder grain structure grain IDs for top of layer 0 if needed (i.e, if the powder layer height is
        // more than zero cells)
        if (powder_top_z > powder_bottom_z) {
            int size_z = powder_top_z - powder_bottom_z;
            initVoronoiGrainID(id, grid, powder_bottom_z, size_z, _inputs.powder_grain_spacing, rng_seed,
                               "powder layer 0");
        }
        // Check to make sure enough grain ID values exist to avoid integer overflow in initialization of future powder
        // layers
        const int size_z_all_powder_layers = Kokkos::round((grid.z_max - _inputs.baseplate_top_z) / grid.deltax) + 1;
        const int grain_structure_volume = grid.nx * grid.ny * size_z_all_powder_layers;
        const std::string powder_layers_label = std::to_string(grid._inputs.number_of_layers) + " powder layers";
        const int number_total_powder_grains =
            getNumberOfGrains(grid, grain_structure_volume, _inputs.powder_grain_spacing, powder_layers_label);
        if (id == 0)
            std::cout << "Expected number of grains in the powder layers of this simulation: "
                      << number_total_powder_grains << std::endl;

        // Resize view and init cell types to solid
        Kokkos::realloc(cell_type, grid.domain_size);
        Kokkos::deep_copy(cell_type, Solid);
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            std::cout << "Grain struct initialized" << std::endl;
        }
    }

    // Determine the height of the baseplate, in CA cells
    // The baseplate always starts at the simulation bottom (Z coordinate corresponding to z_min, Z index = 0),
    // regardless of whether the first layer melts the cells at the bottom or not. If baseplate_through_powder is true,
    // the baseplate microstructure extends through the entire simulation domain in Z (size nz). If
    // baseplate_through_powder is false, the baseplate top from the input file is used, or it is assumed that the top
    // of the baseplate is at Z = 0 microns
    int getBaseplateSizeZ(const int id, const Grid &grid) {
        int baseplate_size_z;
        if (_inputs.baseplate_through_powder)
            baseplate_size_z = grid.nz;
        else {
            baseplate_size_z = Kokkos::round((_inputs.baseplate_top_z - grid.z_min) / grid.deltax) + 1;
            int max_baseplate_size_z = Kokkos::round((grid.z_max_layer[0] - grid.z_min) / grid.deltax) + 1;
            if (baseplate_size_z > max_baseplate_size_z) {
                baseplate_size_z = max_baseplate_size_z;
                if (id == 0)
                    std::cout << "Warning: The specified location of the baseplate top is located above the "
                                 "temperature data for the first layer of the problem; it will be truncated to stop at "
                                 "the top Z coordinate of the first layer's temperature data"
                              << std::endl;
            }
        }
        if ((id == 0) && (baseplate_size_z < 1))
            std::cout << "Warning: no baseplate microstructure will be used, as the temperature data does not overlap "
                         "with the input value for the baseplate top"
                      << std::endl;
        return baseplate_size_z;
    }

    // Check that the substrate bounds span the simulation domain in the requested dimension, within some tolerance
    void checkSubstrateBound(const double substrate_bound_low, const int substrate_num_points,
                             const double substrate_cell_spacing, const double simulation_bound_low,
                             const double simulation_bound_high, std::string dim, const double tolerance) {
        if (simulation_bound_low < substrate_bound_low - tolerance) {
            std::string error = "Error: lower bound of simulation in " + dim + " ( " +
                                std::to_string(simulation_bound_low) +
                                ") extends beyond the lower bound of the substrate grain structure from the file (" +
                                std::to_string(substrate_bound_low) + ")";
            throw std::runtime_error(error);
        }
        const double substrate_bound_high =
            substrate_bound_low + static_cast<double>(substrate_num_points - 1) * substrate_cell_spacing;
        if (simulation_bound_high > substrate_bound_high + tolerance) {
            std::string error = "Error: upper bound of simulation in " + dim + " ( " +
                                std::to_string(simulation_bound_high) +
                                ") extends beyond the upper bound of the substrate grain structure from the file (" +
                                std::to_string(substrate_bound_high) + ")";
            throw std::runtime_error(error);
        }
    }

    // Initializes Grain ID values where the substrate comes from a file
    void initBaseplateGrainID(const int id, const Grid &grid, const int baseplate_size_z) {

        // Parse substrate input file
        std::ifstream substrate;
        substrate.open(_inputs.substrate_filename);
        // Ignore first two header lines
        skipLines(substrate, 2);

        // Is this data binary or ASCII
        std::string read_line;
        getline(substrate, read_line);
        const bool binary_s = (read_line.find("BINARY") != std::string::npos);

        // Ignore line
        skipLines(substrate, 1);

        // Get nx_s, ny_s, and nz_s
        std::vector<std::string> dims_read = splitString(substrate);
        const int nx_s = getInputInt(dims_read[1]);
        const int ny_s = getInputInt(dims_read[2]);
        const int nz_s = getInputInt(dims_read[3]);

        // Get origin location
        std::vector<std::string> org_read = splitString(substrate);
        const double x_min_s = getInputDouble(org_read[1]);
        const double y_min_s = getInputDouble(org_read[2]);
        const double z_min_s = getInputDouble(org_read[3]);

        // Get voxel spacing
        std::vector<std::string> vox_spacing_read = splitString(substrate);
        const double deltax_s = getInputDouble(vox_spacing_read[1]);
        if ((deltax_s != getInputDouble(vox_spacing_read[2])) || (deltax_s != getInputDouble(vox_spacing_read[3])))
            throw std::runtime_error("Error: substrate data must have same spacing in all directions");

        // Ensure substrate dimensions can sufficiently cover the solidification domain
        if (id == 0)
            std::cout << "Substrate dimensions from file are " << nx_s << " by " << ny_s << " by " << nz_s
                      << ", voxel spacing is " << deltax_s << std::endl;
        checkSubstrateBound(x_min_s, nx_s, deltax_s, grid.x_min, grid.x_max, "X", grid.deltax);
        checkSubstrateBound(y_min_s, ny_s, deltax_s, grid.y_min, grid.y_max, "Y", grid.deltax);
        checkSubstrateBound(z_min_s, nz_s, deltax_s, grid.z_min, _inputs.baseplate_top_z, "Z", grid.deltax);

        // Ignore line
        skipLines(substrate, 1);

        // Ensure data is of type integer
        getline(substrate, read_line);
        if (!read_line.find("int"))
            throw std::runtime_error("Error: substrate grain ID data must be type int");

        // Ignore line
        skipLines(substrate, 1);

        // Read grain ID data from file into the host view
        Kokkos::View<int ***, Kokkos::HostSpace> grain_id_s_host(Kokkos::ViewAllocateWithoutInitializing("GrainID_S"),
                                                                 nz_s, nx_s, ny_s);
        if (binary_s)
            grain_id_s_host =
                readBinaryField<Kokkos::View<int ***, Kokkos::HostSpace>, int>(substrate, nx_s, ny_s, nz_s, "GrainID");
        else
            grain_id_s_host =
                readASCIIField<Kokkos::View<int ***, Kokkos::HostSpace>>(substrate, nx_s, ny_s, nz_s, "GrainID");
        if (id == 0)
            std::cout << "Successfully read substrate GrainID data from the file" << std::endl;
        substrate.close();
        // Copy host view data to device
        auto grain_id_s = Kokkos::create_mirror_view_and_copy(memory_space(), grain_id_s_host);

        // Assign each CA cell the GrainID of the nearest voxel in the substrate, returning the min and max GrainID
        const int baseplate_top_coord_1D = grid.nx * grid.ny_local * baseplate_size_z;
        // Struct to store reduction result
        Kokkos::MinMax<int>::value_type bounds_grain_id_local;
        Kokkos::MinMax<int> bounds_grain_id_reducer(bounds_grain_id_local);
        // Local copy for lambda capture
        auto grain_id_all_layers_local = grain_id_all_layers;
        auto policy = Kokkos::RangePolicy<execution_space>(0, baseplate_top_coord_1D);
        Kokkos::parallel_reduce(
            "BaseplateInit", policy,
            KOKKOS_LAMBDA(const int &index_all_layers, Kokkos::MinMax<int>::value_type &update) {
                // x, y, z associated with this 1D cell location, with respect to the global simulation bounds
                const int coord_x_global = grid.getCoordX(index_all_layers);
                const int coord_y_global = grid.getCoordY(index_all_layers) + grid.y_offset;
                const int coord_z_global = grid.getCoordZ(index_all_layers);
                const double x_location = grid.x_min + coord_x_global * grid.deltax;
                const double y_location = grid.y_min + coord_y_global * grid.deltax;
                const double z_location = grid.z_min + coord_z_global * grid.deltax;
                // What voxel does this correspond to in the substrate?
                const int coord_x_s = Kokkos::round((x_location - x_min_s) / deltax_s);
                const int coord_y_s = Kokkos::round((y_location - y_min_s) / deltax_s);
                const int coord_z_s = Kokkos::round((z_location - z_min_s) / deltax_s);
                grain_id_all_layers_local(index_all_layers) = grain_id_s(coord_z_s, coord_x_s, coord_y_s);
                Kokkos::MinMaxScalar<int> current{grain_id_all_layers_local(index_all_layers),
                                                  grain_id_all_layers_local(index_all_layers)};
                bounds_grain_id_reducer.join(update, current);
            },
            bounds_grain_id_reducer);
        Kokkos::fence();
        // Avoid reusing grain IDs that were already in the baseplate grain structure in future powder layers (positive
        // values) or future nucleation events (negative values)
        const int min_grain_id_local = bounds_grain_id_local.min_val;
        const int max_grain_id_local = bounds_grain_id_local.max_val;
        MPI_Allreduce(&min_grain_id_local, &num_prior_nuclei, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_grain_id_local, &next_layer_first_epitaxial_grain_id, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // First grain ID used in future layers should be 1 beyond the largest used in the substrate, increment by 1
        next_layer_first_epitaxial_grain_id++;
        // The number of negative grain IDs to be reserved is equivalent to the absolute value of the smallest substrate
        // grain ID (this will then be passed to the nucleation constructor to avoid assigning these IDs to new
        // nucleation events)
        num_prior_nuclei = Kokkos::abs(num_prior_nuclei);
        if (id == 0)
            std::cout << "Substrate file read complete: grain id values in baseplate: " << -num_prior_nuclei
                      << " through " << next_layer_first_epitaxial_grain_id - 1 << std::endl;
    }

    // Given a size in the Z direction (could be a baseplate or layer height) and a grain spacing (could be for the
    // baseplate or for a powder layer), determine the number of grains expected to be initialized in the region. Throw
    // an error if the number of grains exceeds the maximum integer
    int getNumberOfGrains(const Grid &grid, const int grain_structure_volume, const double mean_grain_spacing,
                          std::string region_type) {
        const double grain_structure_volume_microns =
            grain_structure_volume * Kokkos::pow(grid.deltax, 3) * Kokkos::pow(10, 18);
        const double mean_grain_volume_microns = Kokkos::pow(mean_grain_spacing, 3);
        // Kokkos::lround not provided, using std function here instead
        long int number_of_grains_long = std::lround(grain_structure_volume_microns / mean_grain_volume_microns);
        if (number_of_grains_long > INT_MAX) {
            std::string error = "Error: Input mean grain size used for " + region_type +
                                " will lead to potential integer overflow, to avoid this, a smaller mean baseplate "
                                "and/or powder grain size should be used";
            throw std::runtime_error(error);
        }
        // Need at least 1 baseplate/powder grain, cannot have more baseplate/powder grains than cells
        int number_of_grains = static_cast<int>(number_of_grains_long);
        number_of_grains = std::max(number_of_grains, 1);
        number_of_grains = std::min(number_of_grains, grain_structure_volume);
        return number_of_grains;
    }

    // Initializes Grain ID values using a grain spacing and a Voronoi Tessellation (size_z = baseplate_size_z for
    // initialization of initial substrate, or layer_height for initialization of powder layer (grain mean_grain_spacing
    // = _inputs.substrate_grain_spacing for initialization of initial substrate or _inputs.powder_grain_spacing for
    // initialization of powder layer)
    void initVoronoiGrainID(const int id, const Grid &grid, const int region_bottom, const int size_z,
                            const double mean_grain_spacing, const unsigned long rng_seed, std::string region_type) {

        std::mt19937_64 gen(rng_seed);

        // Based on the desired volume (convert to cubic microns to match units) and the substrate grain spacing,
        // determine the number of grains
        const int grain_structure_volume = grid.nx * grid.ny * size_z;
        const int number_of_grains = getNumberOfGrains(grid, grain_structure_volume, mean_grain_spacing, region_type);
        // TODO: Use device RNG to generate grain locations, instead of host with copy
        // List of potential grain IDs (starting at next_layer_first_epitaxial_grain_id) - index corresponds to the
        // associated CA cell location Assign positive values for indices 0 through number_of_grains-1, temporarily
        // assign zeros to the remaining indices
        std::vector<int> new_grain_locations(grain_structure_volume);
        std::vector<int> new_grain_ids(grain_structure_volume);
        for (int i = 0; i < grain_structure_volume; i++) {
            new_grain_locations[i] = i;
            if (i < number_of_grains)
                new_grain_ids[i] = next_layer_first_epitaxial_grain_id + i;
            else
                new_grain_ids[i] = 0;
        }
        // Shuffle list of new grain IDs
        std::shuffle(new_grain_ids.begin(), new_grain_ids.end(), gen);

        // Create views of new grain IDs and locations - copying new_grain_ids and new_grain_locations
        // values only for indices where new_grain_ids(i) != 0 (cells with new_grain_ids(i) = 0 were not
        // assigned baseplate center locations)
        view_type_int_host new_grain_locations_host(Kokkos::ViewAllocateWithoutInitializing("new_grain_locations_Host"),
                                                    number_of_grains);
        view_type_int_host new_grain_ids_host(Kokkos::ViewAllocateWithoutInitializing("new_grain_ids_Host"),
                                              number_of_grains);
        int count = 0;
        for (int i = 0; i < grain_structure_volume; i++) {
            if (new_grain_ids[i] != 0) {
                new_grain_locations_host(count) = new_grain_locations[i];
                new_grain_ids_host(count) = new_grain_ids[i];
                count++;
            }
        }

        // Copy new grain views to the device
        auto new_grain_ids_device = Kokkos::create_mirror_view_and_copy(memory_space(), new_grain_ids_host);
        auto new_grain_locations_device = Kokkos::create_mirror_view_and_copy(memory_space(), new_grain_locations_host);
        const int region_top = region_bottom + size_z;
        if (id == 0) {
            std::cout << region_type << " spanning domain coordinates Z = " << region_bottom << " through "
                      << region_top - 1 << std::endl;
            std::cout << "Number of grains: " << number_of_grains << ", spanning GrainID "
                      << next_layer_first_epitaxial_grain_id << " through "
                      << next_layer_first_epitaxial_grain_id + number_of_grains - 1 << std::endl;
        }

        // Local copies for lambda capture.
        auto grain_id_all_layers_local = grain_id_all_layers;

        // First, assign cells that are associated with grain centers the appropriate non-zero GrainID values (assumes
        // GrainID values were initialized to zeros)
        auto policy = Kokkos::RangePolicy<execution_space>(0, number_of_grains);
        Kokkos::parallel_for(
            "GrainCenterInit", policy, KOKKOS_LAMBDA(const int &n) {
                int new_grain_loc = grid.nx * grid.ny * region_bottom + new_grain_locations_device(n);
                // x, y, z associated with baseplate grain "n", at 1D coordinate "new_grain_loc"
                int coord_z_all_layers = grid.getCoordZGlobal(new_grain_loc);
                int coord_y_global = grid.getCoordYGlobal(new_grain_loc);
                int coord_x = grid.getCoordXGlobal(new_grain_loc);
                if ((coord_y_global >= grid.y_offset) && (coord_y_global < grid.y_offset + grid.ny_local)) {
                    // This grain is associated with a cell on this MPI rank
                    int coord_y = coord_y_global - grid.y_offset;
                    int index_all_layers = grid.get1DIndex(coord_x, coord_y, coord_z_all_layers);
                    grain_id_all_layers_local(index_all_layers) = new_grain_ids_device(n);
                }
            });
        Kokkos::fence();

        auto md_policy =
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                {region_bottom, 0, 0}, {region_top, grid.nx, grid.ny_local});
        // For cells that are not associated with grain centers, assign them the GrainID of the nearest grain center
        Kokkos::parallel_for(
            "VoronoiGrainAssignment", md_policy,
            KOKKOS_LAMBDA(const int coord_z_all_layers, const int coord_x, const int coord_y) {
                int index_all_layers = grid.get1DIndex(coord_x, coord_y, coord_z_all_layers);
                if (grain_id_all_layers_local(index_all_layers) == 0) {
                    // This cell needs to be assigned a GrainID value
                    // Check each possible baseplate grain center to find the closest one
                    float min_distance_to_this_grain = static_cast<float>(grain_structure_volume);
                    int min_distance_to_this_grain_grain_id = 0;
                    for (int n = 0; n < number_of_grains; n++) {
                        // Baseplate grain center at x_n, y_n, z_n - how far is the cell at i, j+y_offset, k?
                        int coord_z_grain_all_layers =
                            region_bottom + grid.getCoordZGlobal(new_grain_locations_device(n));
                        int coord_y_grain_global = grid.getCoordYGlobal(new_grain_locations_device(n));
                        int coord_x_grain = grid.getCoordXGlobal(new_grain_locations_device(n));
                        int coord_y_global = coord_y + grid.y_offset;
                        float distance_to_this_grain_x = coord_x - coord_x_grain;
                        float distance_to_this_grain_y = coord_y_global - coord_y_grain_global;
                        float distance_to_this_grain_z = coord_z_grain_all_layers - coord_z_all_layers;
                        float distance_to_this_grain =
                            Kokkos::hypot(distance_to_this_grain_x, distance_to_this_grain_y, distance_to_this_grain_z);
                        if (distance_to_this_grain < min_distance_to_this_grain) {
                            // This is the closest grain center to cell at "CAGridLocation" - update values
                            min_distance_to_this_grain = distance_to_this_grain;
                            min_distance_to_this_grain_grain_id = new_grain_ids_device(n);
                        }
                    }
                    // GrainID associated with the closest baseplate grain center
                    grain_id_all_layers_local(index_all_layers) = min_distance_to_this_grain_grain_id;
                }
            });

        // Set grain ID for next layer to avoid reusing any previously assigned values
        next_layer_first_epitaxial_grain_id += number_of_grains;
        if (id == 0)
            std::cout << region_type << " grain structure initialized, number of total grain ID values reserved: "
                      << next_layer_first_epitaxial_grain_id - 1 << std::endl;
    }

    // Sets up views, powder layer (if necessary), and cell types for the next layer of a multilayer problem
    void initNextLayer(const int nextlayernumber, const int id, const Grid &grid, const unsigned long rng_seed) {

        // Subviews for the next layer's grain id, layer id, cell type are constructed based on updated layer bound
        // z_layer_bottom
        // Powder layer extends from Z = powder_bottom_z (1 cell above the top of the previous layer) up to but not
        // including Z = powder_top_z
        int powder_bottom_z = Kokkos::round((grid.z_max_layer[nextlayernumber - 1] - grid.z_min) / grid.deltax) + 1;
        int powder_top_z = Kokkos::round((grid.z_max_layer[nextlayernumber] - grid.z_min) / grid.deltax) + 1;
        if (!(_inputs.baseplate_through_powder)) {
            // Unique RNG seed for each powder layer to avoid placing powder grains in same locations
            std::string powder_layer_label = "powder layer " + std::to_string(nextlayernumber);
            const unsigned long powder_rng_seed = rng_seed + static_cast<unsigned long>(nextlayernumber);
            initVoronoiGrainID(id, grid, powder_bottom_z, powder_top_z - powder_bottom_z, _inputs.powder_grain_spacing,
                               powder_rng_seed, powder_layer_label);
        }

        // Resize and init cell types to Solid for layer "layernumber + 1"
        Kokkos::realloc(cell_type, grid.domain_size);
        Kokkos::deep_copy(cell_type, Solid);
        // Current layer melt pool edge indicator, if needed
        if (_store_melt_pool_edge)
            getCurrentLayerMeltEdge(grid.layer_range);
    }

    // Stores/returns the volume fraction of nucleated grains to the console
    // Moved from CAfunctions.hpp
    float calcVolFractionNucleated(const int id, const Grid &grid, Temperature<memory_space> &temperature) {

        // For interior cells, add the number of cells that underwent melting/solidification and the number of cells
        // with sub-zero grain IDs
        int melted_cells_local = 0;
        int nucleated_grain_cells_local = 0;
        // Local copies for lambda capture.
        auto grain_id_all_layers_local = grain_id_all_layers;
        Kokkos::parallel_reduce(
            "NumSolidifiedCells", grid.domain_size_all_layers,
            KOKKOS_LAMBDA(const int index, int &update_meltcount, int &update_nucleatecount) {
                int coord_y = grid.getCoordY(index);
                // Is this Y coordinate in the halo region? If so, do not increment counter
                bool in_halo_region = false;
                if (((coord_y == 0) && (!grid.at_south_boundary)) ||
                    ((coord_y == grid.ny_local - 1) && (!grid.at_north_boundary)))
                    in_halo_region = true;
                if ((grain_id_all_layers_local(index) < 0) && (!in_halo_region))
                    update_nucleatecount++;
                if ((temperature.layer_id_all_layers(index) != -1) && (!in_halo_region))
                    update_meltcount++;
            },
            melted_cells_local, nucleated_grain_cells_local);

        // Reduce the values by summing over all ranks
        int melted_cells_global, nucleated_grain_cells_global;
        MPI_Allreduce(&melted_cells_local, &melted_cells_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nucleated_grain_cells_local, &nucleated_grain_cells_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Calculate nucleated grain fraction
        float vol_fraction_nucleated =
            static_cast<float>(nucleated_grain_cells_global) / static_cast<float>(melted_cells_global);
        if (id == 0)
            std::cout << "The fraction of the solidified material consisting of nucleated grains is "
                      << vol_fraction_nucleated << std::endl;
        return vol_fraction_nucleated;
    }

    // If cell is liquid, undercooling at solidification start is 0. If cell is active or solid, use stored value from
    // temperature view
    void calcUndercoolingStart(const int domain_size, Temperature<memory_space> &temperature) {

        auto _cell_type = cell_type;
        Kokkos::parallel_for(
            "calcUndercoolingStart", domain_size, KOKKOS_LAMBDA(const int &index) {
                if (_cell_type(index) == Liquid)
                    temperature.undercooling_solidification_start(index) = 0.0;
            });
    }

    // If cell is liquid or active, undercooling is 0 if superheated, otherwise calculated from the cooling
    // rate/liquidus time. If cell is solid, use stored value
    void calcUndercoolingCurrent(const int domain_size, Temperature<memory_space> &temperature, const int cycle = 0) {

        auto _cell_type = cell_type;
        Kokkos::parallel_for(
            "calcUndercoolingCurrent", domain_size, KOKKOS_LAMBDA(const int &index) {
                if (_cell_type(index) != Solid) {
                    if (temperature.current_cooling_rate(index) >= 0)
                        temperature.undercooling_current(index) = temperature.getUndercooling(cycle, index);
                }
            });
    }

    // If storing whether or not a cell is at a melt pool edge, update the value
    KOKKOS_INLINE_FUNCTION
    void setMeltEdge(const int index, const bool updated_state) const {
        if (_store_melt_pool_edge)
            melt_edge(index) = updated_state;
    }

    // Take a view consisting of data for all layers, and return a subview of the same type consisting of just the cells
    // corresponding to the current layer of a multilayer problem
    auto getGrainIDSubview(const Grid &grid) const { return Kokkos::subview(grain_id_all_layers, grid.layer_range); }
    auto getPhaseIDSubview(const Grid &grid) const { return Kokkos::subview(phase_id_all_layers, grid.layer_range); }
};

#endif
