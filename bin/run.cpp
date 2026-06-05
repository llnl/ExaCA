// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "ExaCA.hpp"

#include <Kokkos_Core.hpp>

#include "mpi.h"

#include <stdexcept>
#include <string>

int main(int argc, char *argv[]) {
    // Initialize MPI
    int id, np;
    MPI_Init(&argc, &argv);
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

        // Get number of processes
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        // Get individual process ID
        MPI_Comm_rank(MPI_COMM_WORLD, &id);

        if (id == 0) {
            std::cout << "ExaCA version: " << version() << " \nExaCA commit:  " << gitCommitHash()
                      << "\nKokkos version: " << kokkosVersion() << std::endl;
            Kokkos::DefaultExecutionSpace().print_configuration(std::cout);
            std::cout << "Number of MPI ranks = " << np << std::endl;
        }
        if (argc < 2) {
            throw std::runtime_error("Error: Must provide path to input file on the command line.");
        }
        else {

            // Create timers
            Timers timers(id);
            timers.startInit();

            // Run CA code using reduced temperature data format
            std::string input_file = argv[1];
            // Read input file
            Inputs inputs(id, input_file);
            std::string simulation_type = inputs.simulation_type;
            // Full domain solidification - all cells initially liquid or active and end up solid
            bool full_domain_solidification;
            if ((simulation_type == "Directional") || (simulation_type == "SingleGrain"))
                full_domain_solidification = true;
            else
                full_domain_solidification = false;

            // Setup local and global grids, decomposing domain (needed to construct temperature)
            Grid grid(inputs.simulation_type, id, np, inputs.domain.number_of_layers, inputs.domain, inputs.substrate,
                      inputs.temperature);
            // Temperature fields characterized by data in this structure
            Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);

            // Material response function
            InterfacialResponseFunction irf(inputs.domain.deltat, grid.deltax, inputs.irf);

            // Read temperature data if necessary. For the spot problem, store spot melt data as if it were read from a
            // file
            if (simulation_type == "FromFile")
                temperature.readTemperatureData(id, grid, 0);
            else if (simulation_type == "Spot")
                temperature.storeSpotData(id, grid, irf.freezingRange(), inputs.domain.deltat,
                                          inputs.domain.spot_radius);

            // Initialize the temperature fields for the simulation type of interest. These are either simple
            // unidirectional fields, or more complex data stored in a view in the temperature struct
            if (full_domain_solidification)
                temperature.initialize(id, simulation_type, grid, inputs.domain.deltat);
            else
                temperature.initialize(0, id, grid, irf.freezingRange(), inputs.domain.deltat);
            MPI_Barrier(MPI_COMM_WORLD);

            // Initialize grain orientations
            Orientation<memory_space> orientation(id, inputs.crystal_orientation_file, false, inputs.rng_seed,
                                                  inputs.irf.num_phases, irf.solidificationTransformation());
            MPI_Barrier(MPI_COMM_WORLD);

            // Initialize cell types, grain IDs, and layer IDs
            CellData<memory_space> celldata(grid, inputs.substrate, inputs.print.store_melt_pool_edge);
            if (simulation_type == "Directional")
                celldata.initSubstrate_Directional(id, grid, inputs.rng_seed);
            else if (simulation_type == "SingleGrain")
                celldata.initSubstrate_SingleGrain(id, grid);
            else
                celldata.initSubstrate_BaseplatePowder(id, grid, inputs.rng_seed);
            MPI_Barrier(MPI_COMM_WORLD);

            // Variables characterizing the active cell region within each rank's grid, including buffers for ghost node
            // data (fixed size) and the steering vector/steering vector size on host/device
            Interface<memory_space> interface(id, grid.domain_size, inputs.substrate.init_oct_size);
            // Initialize octahedra for initial active cells, if necessary for this problem type
            if (full_domain_solidification)
                createOctahedra_NoRemelt(grid, celldata, temperature, orientation, interface);
            MPI_Barrier(MPI_COMM_WORLD);

            // Nucleation data structure, containing views of nuclei locations, time steps, and ids, and nucleation
            // event counters - initialized with an estimate on the number of nuclei in the layer Without knowing
            // estimated_nuclei_this_rank_this_layer yet, initialize nucleation data structures to estimated sizes,
            // resize inside of placeNuclei when the number of nuclei per rank is known
            int estimated_nuclei_this_rank_this_layer =
                inputs.nucleation.n_max * pow(grid.deltax, 3) * grid.domain_size;
            Nucleation<memory_space> nucleation(estimated_nuclei_this_rank_this_layer, inputs.nucleation,
                                                celldata.num_prior_nuclei);
            // Fill in nucleation data structures, and assign nucleation undercooling values to potential nucleation
            // events Potential nucleation grains are only associated with liquid cells in layer 0 - they will be
            // initialized for each successive layer when layer 0 is complete
            nucleation.placeNuclei(simulation_type, temperature, irf, inputs.rng_seed, 0, grid, id,
                                   inputs.domain.deltat);

            // Initialize printing struct from inputs
            Print print(grid, np, inputs.print);

            // End of initialization
            timers.stopInit();
            MPI_Barrier(MPI_COMM_WORLD);

            int cycle = 0;
            timers.startRun();

            // Run ExaCA to model solidification of each layer
            for (int layernumber = 0; layernumber < grid.number_of_layers; layernumber++) {
                timers.startLayer();
                runExaCALayer(id, np, layernumber, cycle, inputs, timers, grid, temperature, irf, orientation, celldata,
                              interface, nucleation, print, simulation_type, full_domain_solidification);

                if (layernumber != grid.number_of_layers - 1) {
                    // Initialize new temperature field data for layer "layernumber + 1"
                    // TODO: reorganize these temperature functions calls into a temperature.init_next_layer as done
                    // with the substrate If the next layer's temperature data isn't already stored, it should be read
                    if ((simulation_type == "FromFile") && (inputs.temperature.layerwise_temp_read))
                        temperature.readTemperatureData(id, grid, layernumber + 1);
                    MPI_Barrier(MPI_COMM_WORLD);

                    // Initialize next layer of the simulation
                    initExaCALayer(id, layernumber, cycle, simulation_type, inputs, grid, irf, temperature, celldata,
                                   interface, nucleation);
                    MPI_Barrier(MPI_COMM_WORLD);
                    timers.stopLayer(layernumber);
                }
                else {
                    MPI_Barrier(MPI_COMM_WORLD);
                    timers.stopLayer();
                }
            }
            timers.stopRun();
            MPI_Barrier(MPI_COMM_WORLD);

            // Print ExaCA end-of-run data
            finalizeExaCA(id, np, cycle, inputs, timers, grid, temperature, orientation, celldata, interface, print);
        }
    }
    // Finalize Kokkos
    Kokkos::finalize();
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
