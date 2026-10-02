// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAcelldata.hpp"
#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAinterface.hpp"
#include "CAnucleation.hpp"
#include "CAparsefiles.hpp"
#include "CAtemperature.hpp"
#include "CAtypes.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// tests for Nucleation struct
//---------------------------------------------------------------------------//
// Tests Nucleation object construction and placeNuclei
void testNucleiInit() {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    // default inputs struct
    Inputs inputs;
    inputs.domain.deltat = 1.0;
    // manually set grid for test of 2 layer problem
    int number_of_layers_temp = 2;
    // Initializing layer 1
    int layernumber = 1;
    Grid grid(number_of_layers_temp);
    // Create test data
    // Only Z = 1 through 4 is part of this layer
    grid.nz_layer = 4;
    grid.z_layer_bottom = 1;
    grid.nx = 4;
    // Each rank is assigned a different portion of the domain in Y
    grid.ny = 2 * np;
    grid.ny_local = 2;
    grid.y_offset = 2 * id;
    grid.nz = 6;
    grid.deltax = 1;
    // Global grid XYZ bounds for the layer of interest (layer 1)
    grid.x_min = 0.0;
    grid.x_max = grid.nx * grid.deltax;
    grid.y_min = 0.0;
    grid.y_max = grid.ny * grid.deltax;
    grid.z_min_layer[1] = grid.z_layer_bottom * grid.deltax;
    grid.z_max_layer[1] = (grid.z_layer_bottom + grid.nz_layer - 1) * grid.deltax;
    grid.layer_height = 1;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz_layer;
    grid.domain_size_all_layers = grid.nx * grid.ny_local * grid.nz;
    grid.bottom_of_current_layer = grid.getBottomOfCurrentLayer();
    grid.top_of_current_layer = grid.getTopOfCurrentLayer();
    grid.layer_range = std::make_pair(grid.bottom_of_current_layer, grid.top_of_current_layer);
    // MPI rank locations relative to the global grid
    if (id == 0)
        grid.at_south_boundary = true;
    else
        grid.at_south_boundary = false;
    if (id == np - 1)
        grid.at_north_boundary = true;
    else
        grid.at_north_boundary = false;

    // Simple IRF - single phase
    inputs.irf.num_phases = 1;
    inputs.irf.A[0] = 0.0;
    inputs.irf.B[0] = 0.0;
    inputs.irf.C[0] = 0.0;
    inputs.irf.D[0] = 0.01;
    inputs.irf.function[0] = inputs.irf.cubic;
    InterfacialResponseFunction irf(inputs.domain.deltat, grid.deltax, inputs.irf);

    // There are 40 * np total cells in this domain (nx * ny * nz)
    // Each rank has 40 cells - the top 32 cells are part of the active layer and are candidates for nucleation
    // assignment
    // A cell can solidify 1-3 times
    int max_solidification_events_count = 3;
    // Manually set nucleation parameters
    // This nucleation density ensures there will be 4 potential nuclei per MPI rank present
    // without remelting (each cell solidifies once)
    inputs.nucleation.dtn = 1;
    inputs.nucleation.dtsigma = 0.0001;
    inputs.nucleation.n_max = 0.125;

    // Allocate temperature data structures
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    // Init first_value, last_value, and raw_temperature_data
    // Assume 10 events already happened in layer 0
    temperature.first_value(0) = 0;
    temperature.last_value(0) = 10;
    int num_temperature_data_pts = 10;
    for (int n = 0; n < 3; n++) {
        for (int index = 0; index < grid.domain_size; index++) {
            // Cell coordinates on this rank in X, Y, and Z (GlobalZ = relative to domain bottom)
            int coord_z = grid.getCoordZ(index);
            int coord_z_all_layers = coord_z + grid.z_layer_bottom;
            int coord_y = grid.getCoordY(index);
            int coord_y_global = coord_y + grid.y_offset;
            int coord_x = grid.getCoordX(index);
            int num_solidification_events;
            if (coord_x < grid.nx / 2 - 1)
                num_solidification_events = 3;
            else if (coord_x < grid.nx / 2)
                num_solidification_events = 2;
            else
                num_solidification_events = 1;
            if (n < num_solidification_events) {
                temperature.raw_temperature_data(num_temperature_data_pts, 0) = coord_x * grid.deltax;
                temperature.raw_temperature_data(num_temperature_data_pts, 1) = coord_y_global * grid.deltax;
                temperature.raw_temperature_data(num_temperature_data_pts, 2) = coord_z * grid.deltax;
                // melting time step depends on solidification event number
                int melt_time = coord_z_all_layers + coord_y + grid.y_offset + (grid.domain_size * n);
                temperature.raw_temperature_data(num_temperature_data_pts, 3) = melt_time * inputs.domain.deltat;
                // liquidus time stemp depends on solidification event number
                temperature.raw_temperature_data(num_temperature_data_pts, 4) = (melt_time + 1) * inputs.domain.deltat;
                // ensures that a cell's nucleation time will be 1 time step after its CritTimeStep value
                temperature.raw_temperature_data(num_temperature_data_pts, 5) = 1.2;
                num_temperature_data_pts++;
            }
        }
    }
    // Resize with size now known
    Kokkos::resize(temperature.raw_temperature_data, num_temperature_data_pts, 6);
    temperature.first_value(layernumber) = 10;
    temperature.last_value(layernumber) = num_temperature_data_pts;

    // Init temperature data structures (freezing_range value not used in any calculations
    double freezing_range = 20.0;
    temperature.initialize(layernumber, id, grid, freezing_range, inputs.domain.deltat);

    // Nucleation data structure, containing views of nuclei locations, time steps, and ids, and nucleation event
    // counters - initialized with an estimate on the number of nuclei in the layer
    // (estimated_nuclei_this_rank_this_layer)
    const double domain_volume = (grid.x_max - grid.x_min) * (grid.y_max - grid.y_min) *
                                 (grid.z_max_layer(layernumber) - grid.z_min_layer(layernumber)) * pow(grid.deltax, 3);
    int estimated_nuclei_this_rank_this_layer = inputs.nucleation.n_max * pow(grid.deltax, 3) * grid.domain_size;
    Nucleation<memory_space> nucleation(
        estimated_nuclei_this_rank_this_layer, inputs.nucleation,
        100); // nuclei_grain_id should start at -101 - supply optional input arg to constructor
    // Ensure nucleation inputs in nucleation struct were correctly initialized
    EXPECT_DOUBLE_EQ(inputs.nucleation.n_max, nucleation._inputs.n_max);
    EXPECT_DOUBLE_EQ(inputs.nucleation.dtn, nucleation._inputs.dtn);
    EXPECT_DOUBLE_EQ(inputs.nucleation.dtsigma, nucleation._inputs.dtsigma);

    // Fill in nucleation data structures, and assign nucleation undercooling values to potential nucleation events
    // Potential nucleation grains are only associated with liquid cells in layer 1 - they will be initialized for each
    // successive layer when layer 1 in complete
    nucleation.placeNuclei("FromFile", temperature, irf, inputs.rng_seed, 1, grid, id, inputs.domain.deltat);

    // Copy results back to host to check
    auto nuclei_location_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nucleation.nuclei_locations);
    auto nuclei_grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nucleation.nuclei_grain_id);

    // Was the nucleation counter initialized to zero?
    EXPECT_EQ(nucleation.nucleation_counter, 0);

    // Is the total number of nuclei in the system correct, based on the number of remelting events? Equal probability
    // of creating a nucleus each time a cell resolidifies
    const int expected_nuclei_per_rank =
        100 + max_solidification_events_count * inputs.nucleation.n_max * domain_volume;
    EXPECT_EQ(nucleation.nuclei_whole_domain, expected_nuclei_per_rank);
    for (int n = 0; n < nucleation.possible_nuclei; n++) {
        // Are the nuclei grain IDs negative numbers in the expected range based on the inputs?
        EXPECT_GT(nuclei_grain_id_host(n), -(100 + expected_nuclei_per_rank * np + 1));
        EXPECT_LT(nuclei_grain_id_host(n), -100);
        // Are the correct undercooling values associated with the correct cell locations?
        // Cell location is a local position (relative to the bottom of the layer)
        int index = nuclei_location_host(n);
        int coord_z = grid.getCoordZ(index);
        int coord_y = grid.getCoordY(index);
        int coord_z_all_layers = coord_z + grid.z_layer_bottom;
        // Expected nucleation time with remelting can be one of 3 possibilities, depending on the associated
        // solidification event. Should be 1 time step after the liquidus time which in turn is 1 time step after the
        // melting time
        int melt_time = coord_z_all_layers + coord_y + grid.y_offset;
        int expected_nucleation_time_no_rm = melt_time + 2;
        int associated_s_event = nucleation.nucleation_times_host(n) / grid.domain_size;
        // Account for the fact that the melting/liquidus data from the input are increased by 1 time step to avoid
        // nucleation prior to the first time step
        int expected_nucleation_time_rm = expected_nucleation_time_no_rm + associated_s_event * grid.domain_size + 1;
        EXPECT_EQ(nucleation.nucleation_times_host(n), expected_nucleation_time_rm);

        // Are the nucleation events in order of the time steps at which they may occur?
        if (n < nucleation.possible_nuclei - 2) {
            EXPECT_LE(nucleation.nucleation_times_host(n), nucleation.nucleation_times_host(n + 1));
        }
    }
}

// Tests nucleate_grain member function
void testNucleateGrain() {

    using memory_space = TEST_MEMSPACE;
    using view_int = Kokkos::View<int *, TEST_MEMSPACE>;
    using view_int_host = typename view_int::host_mirror_type;

    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // default inputs struct
    Inputs inputs;
    // manually set grid
    Grid grid;
    // Create views - 125 cells, 75 of which are part of the active layer of the domain (Z = 2-5)
    grid.nx = 5;
    grid.ny_local = 5;
    grid.nz = 5;
    grid.z_layer_bottom = 2;
    grid.nz_layer = 3;
    grid.deltax = 1.0;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz_layer;
    grid.domain_size_all_layers = grid.nx * grid.ny_local * grid.nz;
    grid.bottom_of_current_layer = grid.getBottomOfCurrentLayer();
    grid.top_of_current_layer = grid.getTopOfCurrentLayer();
    grid.layer_range = std::make_pair(grid.bottom_of_current_layer, grid.top_of_current_layer);

    // All cells have grain_id of 0, CellType of Liquid - with the exception of the locations where the nucleation
    // events are unable to occur
    CellData<memory_space> celldata(grid, inputs.substrate);
    Kokkos::deep_copy(celldata.cell_type, Liquid);
    auto cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    auto grain_id = celldata.getGrainIDSubview(grid);
    auto grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grain_id);

    // Create test nucleation data - 10 possible events
    int possible_nuclei = 10;
    Nucleation<memory_space> nucleation(possible_nuclei, inputs.nucleation);
    nucleation.possible_nuclei = 10;
    view_int_host nuclei_locations_host(Kokkos::ViewAllocateWithoutInitializing("nuclei_locations_host"),
                                        possible_nuclei);
    view_int_host nuclei_grain_id_host(Kokkos::ViewAllocateWithoutInitializing("nuclei_grain_id_host"),
                                       possible_nuclei);
    for (int n = 0; n < possible_nuclei; n++) {
        // nucleation_times_host values should be in the order in which the events occur - start with setting them
        // between 0 and 9 nuclei_locations are in order starting with 0, through 9 (locations relative to the bottom of
        // the layer)
        nucleation.nucleation_times_host(n) = n;
        nuclei_locations_host(n) = n;
        // Give these nucleation events grain IDs based on their order, starting with -1 and counting down
        nuclei_grain_id_host(n) = -(n + 1);
    }
    // Include the case where 2 potential nucleation events (3 and 4) happen on the same time step - both successful
    // Let nucleation events 3 and 4 both occur on time step 4
    nucleation.nucleation_times_host(3) = nucleation.nucleation_times_host(4);

    // Include the case where a potential nucleation event (2) is unsuccessful (time step 2)
    int unsuccessful_loc_a = 2;
    cell_type_host(unsuccessful_loc_a) = Active;
    grain_id_host(unsuccessful_loc_a) = 1;

    // Include the case where 2 potential nucleation events (5 and 6) happen on the same time step (time step 6) - the
    // first successful, the second unsuccessful
    nucleation.nucleation_times_host(5) = nucleation.nucleation_times_host(6);
    int unsuccessful_loc_c = 6;
    cell_type_host(unsuccessful_loc_c) = Active;
    grain_id_host(unsuccessful_loc_c) = 2;

    // Include the case where 2 potential nucleation events (8 and 9) happen on the same time step (time step 8) - the
    // first unsuccessful, the second successful
    nucleation.nucleation_times_host(8) = nucleation.nucleation_times_host(9);
    int unsuccessful_loc_b = 8;
    cell_type_host(unsuccessful_loc_b) = Active;
    grain_id_host(unsuccessful_loc_b) = 3;

    // Copy host views to device
    celldata.cell_type = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), cell_type_host);
    grain_id = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), grain_id_host);
    nucleation.nuclei_locations = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), nuclei_locations_host);
    nucleation.nuclei_grain_id = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), nuclei_grain_id_host);

    // Interface struct
    Interface<memory_space> interface(id, grid.domain_size, 0.01, grid.nx, grid.ny);
    // Take enough time steps such that every nucleation event has a chance to occur
    for (int cycle = 0; cycle < 10; cycle++) {
        nucleation.nucleateGrain(cycle, grid, celldata, interface);
    }

    // Copy views back to host to check nucleation results
    cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    auto steering_vector_host_local =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), interface.steering_vector);
    auto num_steer_host_local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), interface.num_steer);
    grain_id = celldata.getGrainIDSubview(grid);
    grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grain_id);

    // Check that all 10 possible nucleation events were attempted
    EXPECT_EQ(nucleation.nucleation_counter, 10);
    // Check that 7 of the 10 nucleation events were successful
    EXPECT_EQ(nucleation.successful_nucleation_counter, 7);
    EXPECT_EQ(num_steer_host_local(0), 7);

    // Ensure that the 3 events that should not have occurred, did not occur
    // These cells should be untouched - active type, and same grain_id they was initialized with
    EXPECT_EQ(cell_type_host(2), Active);
    EXPECT_EQ(grain_id_host(2), 1);
    EXPECT_EQ(cell_type_host(6), Active);
    EXPECT_EQ(grain_id_host(6), 2);
    EXPECT_EQ(cell_type_host(8), Active);
    EXPECT_EQ(grain_id_host(8), 3);

    // Check that the successful nucleation events occurred as expected
    // For each cell location (relative to the layer bottom) that should be home to a successful nucleation event,
    // check that the CellType has been set to FutureActive and the grain_id matches the expected value Also check that
    // the adjusted cell coordinate (relative to the current layer bounds) appears somewhere within the steering vector
    std::vector<int> successful_nuc_grain_ids{-1, -2, -4, -5, -6, -8, -10};
    std::vector<int> successful_nuc_cell_locations{0, 1, 3, 4, 5, 7, 9};
    for (int nevent = 0; nevent < 7; nevent++) {
        int index = successful_nuc_cell_locations[nevent];
        EXPECT_EQ(cell_type_host(index), FutureActive);
        EXPECT_EQ(grain_id_host(index), successful_nuc_grain_ids[nevent]);
        bool on_steering_vector = false;
        for (int svloc = 0; svloc < 7; svloc++) {
            if (steering_vector_host_local(svloc) == index) {
                on_steering_vector = true;
                break;
            }
        }
        EXPECT_TRUE(on_steering_vector);
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, nucleation) {
    testNucleiInit();
    testNucleateGrain();
}
} // end namespace Test
