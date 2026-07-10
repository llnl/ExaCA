// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

#include "CAcelldata.hpp"
#include "CAinputs.hpp"
#include "CAinterface.hpp"
#include "CAparsefiles.hpp"
#include "CAtypes.hpp"
#include "ExaCA.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {

//---------------------------------------------------------------------------//
// full_simulations
//---------------------------------------------------------------------------//
void testSmallDirS() {

    using memory_space = TEST_MEMSPACE;
    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    std::string input_file = "Inp_SmallDirSolidification.json";
    Inputs inputs(id, input_file);
    Timers timers(id);

    // Setup local and global grids, decomposing domain (needed to construct temperature)
    Grid grid(inputs.simulation_type, id, np, inputs.domain.number_of_layers, inputs.domain, inputs.substrate,
              inputs.temperature);
    // Temperature fields characterized by data in this structure
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    temperature.initialize(id, "Directional", grid, inputs.domain.deltat);
    MPI_Barrier(MPI_COMM_WORLD);

    // Run SmallDirS problem and check volume fraction of nucleated grains with 1% tolerance of expected value (to
    // account for the non-deterministic nature of the cell capture)
    // Material response function
    InterfacialResponseFunction irf(inputs.domain.deltat, grid.deltax, inputs.irf);

    // Initialize grain orientations
    Orientation<memory_space> orientation(id, inputs.crystal_orientation_file, false);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize cell types, grain IDs, and layer IDs
    CellData<memory_space> celldata(grid, inputs.substrate, inputs.print.store_melt_pool_edge);
    celldata.initSubstrate_Directional(id, grid, inputs.rng_seed);
    MPI_Barrier(MPI_COMM_WORLD);

    // Variables characterizing the active cell region within each rank's grid, including buffers for ghost node data
    // (fixed size) and the steering vector/steering vector size on host/device
    Interface<memory_space> interface(id, grid.domain_size, inputs.substrate.init_oct_size);
    // Initialize octahedron for active cells
    createOctahedra_NoRemelt(grid, celldata, temperature, orientation, interface);
    MPI_Barrier(MPI_COMM_WORLD);

    // Nucleation data structure, containing views of nuclei locations, time steps, and ids, and nucleation event
    // counters - initialized with an estimate on the number of nuclei in the layer Without knowing
    // estimated_nuclei_this_rank_this_layer yet, initialize nucleation data structures to estimated sizes, resize
    // inside of placeNuclei when the number of nuclei per rank is known
    int estimated_nuclei_this_rank_this_layer = inputs.nucleation.n_max * pow(grid.deltax, 3) * grid.domain_size;
    Nucleation<memory_space> nucleation(estimated_nuclei_this_rank_this_layer, inputs.nucleation);
    // Fill in nucleation data structures, and assign nucleation undercooling values to potential nucleation events
    // Potential nucleation grains are only associated with liquid cells in layer 0 - they will be initialized for each
    // successive layer when layer 0 is complete
    nucleation.placeNuclei("Directional", temperature, irf, inputs.rng_seed, 0, grid, id, inputs.domain.deltat);

    // Initialize printing struct from inputs
    Print print(grid, np, inputs.print);

    // End of initialization
    timers.stopInit();
    MPI_Barrier(MPI_COMM_WORLD);

    int cycle = 0;
    timers.startRun();

    runExaCALayer(id, np, 0, cycle, inputs, timers, grid, temperature, irf, orientation, celldata, interface,
                  nucleation, print, "Directional");

    timers.stopRun();
    MPI_Barrier(MPI_COMM_WORLD);

    // Print ExaCA end-of-run data
    finalizeExaCA(id, np, cycle, inputs, timers, grid, temperature, orientation, celldata, interface, print);
    // MPI barrier to ensure that log file has been written
    MPI_Barrier(MPI_COMM_WORLD);
    std::string log_file = "TestProblemSmallDirS.json";
    std::ifstream log_data_stream(log_file);
    nlohmann::json log_data = nlohmann::json::parse(log_data_stream);
    float vol_fraction_nucleated = log_data["Nucleation"]["VolFractionNucleated"];
    EXPECT_NEAR(vol_fraction_nucleated, 0.1882, 0.0100);
}

void testSmallEquiaxedGrain() {

    using memory_space = TEST_MEMSPACE;
    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    std::string input_file = "Inp_SmallEquiaxedGrain.json";
    Inputs inputs(id, input_file);
    Timers timers(id);

    // Setup local and global grids, decomposing domain (needed to construct temperature)
    Grid grid(inputs.simulation_type, id, np, inputs.domain.number_of_layers, inputs.domain, inputs.substrate,
              inputs.temperature);
    // Temperature fields characterized by data in this structure
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    temperature.initialize(id, "SingleGrain", grid, inputs.domain.deltat);

    // Run Small equiaxed grain problem and check time step at which the grain reaches the domain edge
    // Material response function
    InterfacialResponseFunction irf(inputs.domain.deltat, grid.deltax, inputs.irf);

    // Initialize grain orientations
    Orientation<memory_space> orientation(id, inputs.crystal_orientation_file, false);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize cell types, grain IDs, and layer IDs
    CellData<memory_space> celldata(grid, inputs.substrate, inputs.print.store_melt_pool_edge);
    celldata.initSubstrate_SingleGrain(id, grid);
    MPI_Barrier(MPI_COMM_WORLD);

    // Variables characterizing the active cell region within each rank's grid, including buffers for ghost node data
    // (fixed size) and the steering vector/steering vector size on host/device
    Interface<memory_space> interface(id, grid.domain_size, inputs.substrate.init_oct_size);
    // Initialize octahedron for active cell
    createOctahedra_NoRemelt(grid, celldata, temperature, orientation, interface);
    MPI_Barrier(MPI_COMM_WORLD);

    // Nucleation data structure, containing views of nuclei locations, time steps, and ids, and nucleation event
    // counters - initialized with an estimate on the number of nuclei in the layer Without knowing
    // estimated_nuclei_this_rank_this_layer yet, initialize nucleation data structures to estimated sizes, resize
    // inside of placeNuclei when the number of nuclei per rank is known
    int estimated_nuclei_this_rank_this_layer = 0;
    Nucleation<memory_space> nucleation(estimated_nuclei_this_rank_this_layer, inputs.nucleation);

    // Initialize printing struct from inputs
    Print print(grid, np, inputs.print);

    // End of initialization
    timers.stopInit();
    MPI_Barrier(MPI_COMM_WORLD);

    int cycle = 0;
    timers.startRun();

    runExaCALayer(id, np, 0, cycle, inputs, timers, grid, temperature, irf, orientation, celldata, interface,
                  nucleation, print, "SingleGrain");

    timers.stopRun();
    MPI_Barrier(MPI_COMM_WORLD);

    // Print ExaCA end-of-run data
    finalizeExaCA(id, np, cycle, inputs, timers, grid, temperature, orientation, celldata, interface, print);

    // MPI barrier to ensure that log file has been written
    MPI_Barrier(MPI_COMM_WORLD);
    std::string log_file = "TestProblemSmallEquiaxedGrain.json";
    std::ifstream log_data_stream(log_file);
    nlohmann::json log_data = nlohmann::json::parse(log_data_stream);
    int time_step_of_output = log_data["TimeStepOfOutput"];
    // FIXME: Output time step is usually 4820, but may be 1 time step larger - need to investigate this possible race
    // condition. Finishes 1 time step earlier now after update to active cell init for no remelting case
    EXPECT_NEAR(time_step_of_output, 4820, 1);
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, full_simulations) {
    testSmallDirS();
    testSmallEquiaxedGrain();
}
} // end namespace Test
