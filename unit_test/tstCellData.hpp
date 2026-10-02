// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAcelldata.hpp"
#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAparsefiles.hpp"
#include "CAtypes.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// cell_init_tests
//---------------------------------------------------------------------------//
void testCellDataInit_SingleGrain() {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Default initialized inputs
    Inputs inputs;
    // Let overall domain be 5 cells in X and Z, 50 cells in Y
    // This should in turn place the single grain in the cell at X = Y = 2 and Z = 24 (domain center)
    // Domain for each rank
    inputs.domain.nx = 5;
    inputs.domain.ny = 50;
    inputs.domain.nz = 5;
    int expected_grain_x = Kokkos::floorf(static_cast<float>(inputs.domain.nx) / 2.0);
    int expected_grain_y = Kokkos::floorf(static_cast<float>(inputs.domain.ny) / 2.0);
    int expected_grain_z = Kokkos::floorf(static_cast<float>(inputs.domain.nz) / 2.0);

    // Set up grid and decompose domain
    Grid grid("SingleGrain", id, np, 1, inputs.domain, inputs.substrate, inputs.temperature);

    // Cell data struct
    CellData<memory_space> celldata(grid, inputs.substrate);

    // Check that default substrate single grain orientation was set
    EXPECT_EQ(inputs.substrate.single_grain_orientation, celldata._inputs.single_grain_orientation);

    // Init grain
    celldata.initSubstrate_SingleGrain(id, grid);
    // Copy cell type and grain ID back to host to check if the values match - only 1 cell should've been assigned type
    // active and GrainID = 1 (though it may be duplicated in the ghost nodes of other ranks)
    auto grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.grain_id_all_layers);
    auto cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
        for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
            for (int coord_y = 0; coord_y < grid.ny_local; coord_y++) {
                int coord_y_global = coord_y + grid.y_offset;
                int coord_1d = grid.get1DIndex(coord_x, coord_y, coord_z);
                if ((coord_z == expected_grain_z) && (coord_x == expected_grain_x) &&
                    (coord_y_global == expected_grain_y)) {
                    EXPECT_EQ(grain_id_host(coord_1d), celldata._inputs.single_grain_orientation + 1);
                    EXPECT_EQ(cell_type_host(coord_1d), Active);
                }
                else {
                    EXPECT_EQ(grain_id_host(coord_1d), 0);
                    EXPECT_EQ(cell_type_host(coord_1d), Liquid);
                }
            }
        }
    }
}

void testCellDataInit_Constrained_Automatic(std::string input_surface_init_mode) {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Empty inputs and grid struct
    Inputs inputs;
    Grid grid;
    if (input_surface_init_mode == "SurfaceSiteFraction") {
        // Set quarter of sites active, number of grains will depend on domain size
        inputs.substrate.fract_surface_sites_active = 0.25;
        inputs.substrate.surface_init_mode = "SurfaceSiteFraction";
    }
    else {
        // Set so that there are 4 grains in the domain, regardless of domain size
        inputs.substrate.surface_site_density = 1.0 / (4.0 * static_cast<double>(np));
        inputs.substrate.surface_init_mode = "SurfaceSiteDensity";
    }
    // Create test data
    grid.nz = 2;
    grid.z_layer_bottom = 0;
    grid.nz_layer = 2;
    grid.nx = 4;
    grid.deltax = 1 * pow(10, -6);
    grid.z_max_layer(0) = (grid.nz - 1) * grid.deltax;
    // Domain size in Y depends on the number of ranks - each rank has 4 cells in Y
    // Each rank is assigned a different portion of the domain in Y
    grid.ny = 4 * np;
    grid.ny_local = 4;
    grid.y_offset = 4 * id;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz_layer;
    grid.domain_size_all_layers = grid.nx * grid.ny_local * grid.nz;

    // Construct celldata struct
    CellData<memory_space> celldata(grid, inputs.substrate);
    // Check appropriate initialization of celldata input
    EXPECT_DOUBLE_EQ(inputs.substrate.fract_surface_sites_active, celldata._inputs.fract_surface_sites_active);
    // Initialize substrate grains
    celldata.initSubstrate_Directional(id, np, grid, inputs.rng_seed);
    // Copy CellType, GrainID views to host to check values
    auto cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    auto grain_id_all_layers_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.grain_id_all_layers);
    int max_expected_grain_id;
    if (input_surface_init_mode == "SurfaceSiteFraction")
        max_expected_grain_id = 4 * np + 1;
    else
        max_expected_grain_id = 5;

    for (int index = 0; index < grid.domain_size; index++) {
        if (index >= grid.nx * grid.ny_local) {
            // Not at bottom surface - should be liquid cells with GrainID still equal to 0
            EXPECT_EQ(grain_id_all_layers_host(index), 0);
            EXPECT_EQ(cell_type_host(index), Liquid);
        }
        else {
            // Check that active cells have GrainIDs > 0, and less than max_expected_grain_id (GrainIDs 1 through
            // max_expected_grain_id-1 should've been used)
            if (cell_type_host(index) == Active) {
                EXPECT_GT(grain_id_all_layers_host(index), 0);
                EXPECT_LT(grain_id_all_layers_host(index), max_expected_grain_id);
            }
            else {
                // Liquid cells should still have GrainID = 0
                EXPECT_EQ(grain_id_all_layers_host(index), 0);
            }
        }
    }
}

void testCellDataInit_Constrained_Custom() {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Inputs struct data from TwoGrainDirS example
    Inputs inputs(id, "Inp_TwoGrainDirSolidification.json");

    // Domain size in Y depends on the number of ranks - each rank has 4 cells in Y
    // Each rank is assigned a different portion of the domain in Y
    Grid grid("Directional", id, np, 1, inputs.domain, inputs.substrate, inputs.temperature);

    // Construct celldata struct
    CellData<memory_space> celldata(grid, inputs.substrate);

    // Place substrate grains
    celldata.initSubstrate_Directional(id, np, grid, 0.0);

    // Copy CellType, GrainID views to host to check values
    auto cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    auto grain_id_all_layers_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.grain_id_all_layers);
    // Bottom surface (Z = 0): Check that the two Active cells are in the right locations on the bottom surface,
    // and have the right grain IDs. Otherwise, cells are liquid and have GrainID still equal to 0
    for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
        for (int coord_y_local = 0; coord_y_local < grid.ny_local; coord_y_local++) {
            int index = grid.get1DIndex(coord_x, coord_y_local, 0);
            int coord_y_global = coord_y_local + grid.y_offset;
            if ((coord_x == 100) && (coord_y_global == 50)) {
                EXPECT_EQ(grain_id_all_layers_host(index), 25);
                EXPECT_EQ(cell_type_host(index), Active);
            }
            else if ((coord_x == 100) && (coord_y_global == 150)) {
                EXPECT_EQ(grain_id_all_layers_host(index), 9936);
                EXPECT_EQ(cell_type_host(index), Active);
            }
            else {
                EXPECT_EQ(grain_id_all_layers_host(index), 0);
                EXPECT_EQ(cell_type_host(index), Liquid);
            }
        }
    }
    // Cells at Z > 0: should be liquid cells with GrainID still equal to 0
    for (int index = grid.nx * grid.ny; index < grid.domain_size; index++) {
        EXPECT_EQ(grain_id_all_layers_host(index), 0);
        EXPECT_EQ(cell_type_host(index), Liquid);
    }
}

// Tests substrate init for baseplate and init_next_layer
void testCellDataInit(bool powder_first_layer) {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    // Empty inputs and grid struct for initialization of a problem with 3 layers
    Inputs inputs;
    Grid grid(3);

    // Create test data - 3 layers, starting on layer 0
    grid.deltax = 1 * pow(10, -6);
    grid.nz = 5;

    // First layer: Z = 0 through 2
    grid.nz_layer = 3;
    grid.z_layer_bottom = 0;
    grid.z_min_layer(0) = 0;
    grid.z_max_layer(0) = 2 * grid.deltax;
    grid.z_min = grid.z_min_layer(0);
    // Second layer: Z = 1 through 3
    grid.z_min_layer(1) = 1 * grid.deltax;
    grid.z_max_layer(1) = 3 * grid.deltax;
    // Third layer: Z = 2 through 4
    grid.z_min_layer(2) = 2 * grid.deltax;
    grid.z_max_layer(2) = 4 * grid.deltax;

    grid.nx = 3;
    // Each rank is assigned a different portion of the domain in Y
    grid.ny = 3 * np;
    grid.ny_local = 3;
    grid.y_offset = 3 * id;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz_layer;
    grid.domain_size_all_layers = grid.nx * grid.ny_local * grid.nz;

    // If there is a powder layer, the baseplate should be Z = 0 through 1 w/ powder for the top row of cells, otherwise
    // it should be 0 through 2
    // If there is no powder layer, the baseplate should be Z = 0 through 2 with no powder
    // Manually set non-default substrate values from inputs
    int baseplate_size, expected_num_powder_grains_per_layer;
    if (powder_first_layer) {
        inputs.substrate.baseplate_top_z = grid.deltax;
        baseplate_size = grid.nx * grid.ny_local * (round((grid.z_max_layer(0) - grid.z_min) / grid.deltax));
        expected_num_powder_grains_per_layer = grid.nx * grid.ny_local * np;
    }
    else {
        inputs.substrate.baseplate_top_z = 2 * grid.deltax;
        baseplate_size = grid.nx * grid.ny_local * (round((grid.z_max_layer(0) - grid.z_min) / grid.deltax) + 1);
        expected_num_powder_grains_per_layer = 0;
    }
    // There are 45 * np total cells in this domain (nx * ny * nz)
    // Each rank has 45 cells - the bottom 27 cells are assigned baseplate Grain ID values, unless the powder layer of
    // height 1 is used, in which case only the bottom 18 cells should be assigned baseplate Grain ID values. The top
    // cells (Z > 2) are outside the first layer of the domain and are not assigned Grain IDs with the rest of the
    // baseplate. This grain spacing ensures that there will be 1 grain per number of MPI ranks present (larger when
    // powder layer is present as the baseplate will only have a third as many cells)
    if (powder_first_layer)
        inputs.substrate.baseplate_grain_spacing = 2.62;
    else
        inputs.substrate.baseplate_grain_spacing = 3.0;
    inputs.rng_seed = 0.0;

    // Call constructor
    CellData<memory_space> celldata(grid, inputs.substrate);
    // Check that substrate inputs were copied from inputs struct correctly
    EXPECT_DOUBLE_EQ(inputs.substrate.baseplate_top_z, celldata._inputs.baseplate_top_z);
    EXPECT_DOUBLE_EQ(inputs.substrate.baseplate_grain_spacing, celldata._inputs.baseplate_grain_spacing);
    EXPECT_FALSE(celldata._inputs.use_substrate_file);
    EXPECT_FALSE(celldata._inputs.baseplate_through_powder);
    EXPECT_DOUBLE_EQ(celldata._inputs.powder_grain_spacing, grid.deltax * pow(10, 6));
    // Initialize baseplate grain structure
    celldata.initSubstrate_BaseplatePowder(id, grid, inputs.rng_seed);

    // Copy GrainID results back to host to check first layer's initialization
    auto grain_id_all_layers_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.grain_id_all_layers);

    // Baseplate grains - cells should have GrainIDs between 1 and np (inclusive)
    for (int i = 0; i < baseplate_size; i++) {
        EXPECT_GT(grain_id_all_layers_host(i), 0);
        EXPECT_LT(grain_id_all_layers_host(i), np + 1);
    }

    if (powder_first_layer) {
        // Powder grains should have unique Grain ID values larger than 0 and smaller than
        // NextLayer_FirstEpitaxialGrainID
        EXPECT_EQ(celldata.next_layer_first_epitaxial_grain_id, np + 1 + expected_num_powder_grains_per_layer);
        // Powder should only exist at cells corresponding to Z = 2
        int bottom_powder_layer = grid.nx * grid.ny_local * 2;
        int top_powder_layer = grid.nx * grid.ny_local * 3;
        for (int i = bottom_powder_layer; i < top_powder_layer; i++) {
            EXPECT_GT(grain_id_all_layers_host(i), 0);
            EXPECT_LT(grain_id_all_layers_host(i), celldata.next_layer_first_epitaxial_grain_id);
        }
    }
    else {
        // Next unused GrainID should be the number of grains present in the baseplate plus 1 (since GrainID = 0 is not
        // used for any baseplate grains)
        EXPECT_EQ(celldata.next_layer_first_epitaxial_grain_id, np + 1);
    }

    // Copy cell types back to host to check - all cells should be initialized as solid
    auto cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    for (int index = 0; index < grid.domain_size; index++)
        EXPECT_EQ(cell_type_host(index), Solid);
    int previous_layer_first_epitaxial_grain_id = celldata.next_layer_first_epitaxial_grain_id;
    // Initialize the next layer using the same time-temperature history - powder should span cells at Z = 3
    expected_num_powder_grains_per_layer = grid.nx * grid.ny_local * np;
    grid.initNextLayer(id, "FromFile", 1);
    celldata.initNextLayer(1, id, grid, inputs.rng_seed);

    // Copy all grain IDs for all layers back to the host to check that they match
    // and that the powder layer was initialized correctly
    grain_id_all_layers_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.grain_id_all_layers);

    // Check that the right number of powder grains were initialized
    EXPECT_EQ(celldata.next_layer_first_epitaxial_grain_id,
              previous_layer_first_epitaxial_grain_id + expected_num_powder_grains_per_layer);

    // Powder grains should have unique Grain ID values between PreviousLayer_FirstEpitaxialGrainID and
    // NextLayer_FirstEpitaxialGrainID - 1
    int bottom_powder_layer = grid.nx * grid.ny_local * 3;
    int top_powder_layer = grid.nx * grid.ny_local * 4;
    for (int index_all_layers = bottom_powder_layer; index_all_layers < top_powder_layer; index_all_layers++) {
        EXPECT_GT(grain_id_all_layers_host(index_all_layers), previous_layer_first_epitaxial_grain_id - 1);
        EXPECT_LT(grain_id_all_layers_host(index_all_layers), celldata.next_layer_first_epitaxial_grain_id);
    }

    // Subview grain IDs should match the grain IDs overall
    auto grain_id = celldata.getGrainIDSubview(grid);
    auto grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grain_id);
    for (int index = 0; index < grid.domain_size; index++) {
        int index_all_layers = index + grid.z_layer_bottom * grid.nx * grid.ny_local;
        EXPECT_EQ(grain_id_host(index), grain_id_all_layers_host(index_all_layers));
    }

    // Copy cell types back to host to check - should have also initialized as Solid
    cell_type_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), celldata.cell_type);
    for (int index = 0; index < grid.domain_size; index++)
        EXPECT_EQ(cell_type_host(index), Solid);
}

void testCalcVolFractionNucleated() {

    using memory_space = TEST_MEMSPACE;
    using execution_space = typename memory_space::execution_space;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Empty inputs struct
    Inputs inputs;
    // Empty grid struct to be filled manually
    Grid grid;
    // Get neighbor rank process IDs
    if (id == np - 1)
        grid.at_north_boundary = true;
    else
        grid.at_north_boundary = false;
    if (id == 0)
        grid.at_south_boundary = true;
    else
        grid.at_south_boundary = false;

    // Simulation domain
    grid.nx = 3;
    grid.ny_local = 3;
    grid.nz = 3;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz;
    grid.domain_size_all_layers = grid.domain_size;
    // Let all cells except those at Z = 0 have undergone solidification
    // Let the cells at Z = 1 consist of positive grain IDs, and those at Z = 2 of negative grain IDs
    CellData<memory_space> celldata(grid, inputs.substrate);
    // LayerID stored in temperature view
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    auto md_policy =
        Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, 0, 0}, {grid.nz, grid.nx, grid.ny_local});
    Kokkos::parallel_for(
        "VolFractNucleatedInit", md_policy, KOKKOS_LAMBDA(const int coord_z, const int coord_x, const int coord_y) {
            int index = grid.get1DIndex(coord_x, coord_y, coord_z);
            if (coord_z == 0)
                temperature.layer_id_all_layers(index) = -1;
            else
                temperature.layer_id_all_layers(index) = 0;
            if (coord_z == 2)
                celldata.grain_id_all_layers(index) = -1;
            else
                celldata.grain_id_all_layers(index) = 1;
        });
    // Perform calculation and compare to expected value (half of the solidified portion of the domain should consist of
    // nucleated grains, regardless of the number of MPI ranks used)
    float vol_fraction_nucleated = celldata.calcVolFractionNucleated(id, grid, temperature);
    EXPECT_FLOAT_EQ(vol_fraction_nucleated, 0.5);
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, cell_init_tests) {
    testCellDataInit_SingleGrain();
    testCellDataInit_Constrained_Automatic("SurfaceSiteDensity");
    testCellDataInit_Constrained_Automatic("SurfaceSiteFraction");
    testCellDataInit_Constrained_Automatic("FractionSurfaceSitesActive");
    testCellDataInit_Constrained_Custom();
    // For non-constrained solidification problems, test w/ and w/o space left for powder layer
    testCellDataInit(true);
    testCellDataInit(false);
    testCalcVolFractionNucleated();
}
} // end namespace Test
