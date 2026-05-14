// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAparsefiles.hpp"
#include "CAprint.hpp"
#include "CAtemperature.hpp"
#include "CAtypes.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// tests for Temperature struct
//---------------------------------------------------------------------------//
// Tests constructing temperature object in the case of reading data from a file(s)
void testReadTemperatureData(int number_of_layers, bool layerwise_temp_read, bool test_binary_input_read) {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Create test data
    // Default grid, manually set values
    Grid grid(number_of_layers);
    grid.ny_local = 3;
    grid.y_offset = 3 * id; // each col is separated from the others by 3 cells
    grid.y_min = 0.0;
    grid.deltax = 1 * pow(10, -6);
    grid.y_max = (3 * id - 1) * grid.deltax;

    // Domain size is a 3 by 12 by 3 region
    grid.x_min = 0.0;
    grid.nx = 3;
    grid.x_max = (grid.nx - 1) * grid.deltax;
    grid.ny = 12;
    grid.nz = 3;
    grid.domain_size = grid.nx * grid.ny * grid.nz;
    grid.domain_size_all_layers = grid.domain_size;
    grid.layer_range = std::make_pair(0, grid.domain_size);
    // Write fake OpenFOAM data - only rank 0. Temperature data should be of type double
    // Write two files, one or both of which should be read
    std::string test_temp_filename_1 = "TestData1";
    std::string test_temp_filename_2 = "TestData2";
    if (test_binary_input_read) {
        test_temp_filename_1 = test_temp_filename_1 + ".catemp";
        test_temp_filename_2 = test_temp_filename_2 + ".catemp";
    }
    else {
        test_temp_filename_1 = test_temp_filename_1 + ".txt";
        test_temp_filename_2 = test_temp_filename_2 + ".txt";
    }
    if (id == 0) {
        std::ofstream test_data_file_1;
        if (test_binary_input_read)
            test_data_file_1.open(test_temp_filename_1, std::ios::out | std::ios::binary);
        else {
            test_data_file_1.open(test_temp_filename_1);
            test_data_file_1 << "x, y, z, tm, tl, cr" << std::endl;
        }
        for (int j = 0; j < grid.ny; j++) {
            for (int i = 0; i < grid.nx; i++) {
                if (test_binary_input_read) {
                    writeData(test_data_file_1, static_cast<double>(i * grid.deltax), test_binary_input_read);
                    writeData(test_data_file_1, static_cast<double>(j * grid.deltax), test_binary_input_read);
                    writeData(test_data_file_1, static_cast<double>(0.0), test_binary_input_read);
                    writeData(test_data_file_1, static_cast<double>(i * j), test_binary_input_read);
                    writeData(test_data_file_1, static_cast<double>(i * j + i), test_binary_input_read);
                    writeData(test_data_file_1, static_cast<double>(i * j + j), test_binary_input_read);
                }
                else
                    test_data_file_1 << i * grid.deltax << "," << j * grid.deltax << "," << 0.0 << ","
                                     << static_cast<double>(i * j) << "," << static_cast<double>(i * j + i) << ","
                                     << static_cast<double>(i * j + j) << std::endl;
            }
        }
        test_data_file_1.close();

        std::ofstream test_data_file_2;
        if (test_binary_input_read)
            test_data_file_2.open(test_temp_filename_2, std::ios::out | std::ios::binary);
        else {
            test_data_file_2.open(test_temp_filename_2);
            test_data_file_2 << "x, y, z, tm, tl, cr" << std::endl;
        }
        for (int j = 0; j < grid.ny; j++) {
            for (int i = 0; i < grid.nx; i++) {
                if (test_binary_input_read) {
                    writeData(test_data_file_2, static_cast<double>(i * grid.deltax), test_binary_input_read);
                    writeData(test_data_file_2, static_cast<double>(j * grid.deltax), test_binary_input_read);
                    writeData(test_data_file_2, static_cast<double>(grid.deltax), test_binary_input_read);
                    writeData(test_data_file_2, static_cast<double>(i * j), test_binary_input_read);
                    writeData(test_data_file_2, static_cast<double>(i * j + i), test_binary_input_read);
                    writeData(test_data_file_2, static_cast<double>(i * j + j), test_binary_input_read);
                }
                else
                    test_data_file_2 << i * grid.deltax << "," << j * grid.deltax << "," << grid.deltax << ","
                                     << static_cast<double>(i * j) << "," << static_cast<double>(i * j + i) << ","
                                     << static_cast<double>(i * j + j) << std::endl;
            }
        }
        test_data_file_2.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Test each 12 by 3 subdomain (this test uses MPI - each rank has its own subdomain)
    // Default inputs struct - manually set non-default substrateInputs values
    Inputs inputs;
    inputs.temperature.temp_paths.push_back(test_temp_filename_1);
    inputs.temperature.temp_paths.push_back(test_temp_filename_2);
    inputs.temperature.temp_files_in_series = 2;
    inputs.temperature.layerwise_temp_read = layerwise_temp_read;

    // Ensure that constructor correctly initialized the local values of inputs
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    if (layerwise_temp_read)
        EXPECT_TRUE(temperature._inputs.layerwise_temp_read);
    else
        EXPECT_FALSE(temperature._inputs.layerwise_temp_read);
    EXPECT_EQ(inputs.temperature.temp_files_in_series, temperature._inputs.temp_files_in_series);
    EXPECT_TRUE(temperature._inputs.temp_paths[0] == inputs.temperature.temp_paths[0]);
    EXPECT_TRUE(temperature._inputs.temp_paths[1] == inputs.temperature.temp_paths[1]);

    // Read in data to "raw_temperature_data"
    temperature.readTemperatureData(id, grid, 0);
    // Check the results.
    // Does each rank have the right number of temperature data points? Each rank should have one for each of the 9
    // cells in the subdomain If both files were read, twice as many temperature data points per file should be present
    int number_of_temperature_data_points = temperature.raw_temperature_data.extent(0);
    int num_temp_points_multiplier;
    if (layerwise_temp_read)
        num_temp_points_multiplier = 1;
    else
        num_temp_points_multiplier = std::min(number_of_layers, inputs.temperature.temp_files_in_series);
    EXPECT_EQ(number_of_temperature_data_points, 9 * num_temp_points_multiplier);
    int number_of_cells_per_rank = 9;
    // Does each rank have the right temperature data values?
    for (int layercounter = 0; layercounter < num_temp_points_multiplier; layercounter++) {
        for (int n = 0; n < number_of_cells_per_rank; n++) {
            double expected_values_this_data_point[6];
            // Location on local grid
            int ca_row = n % 3;
            int ca_col = n / 3;
            // X Coordinate
            expected_values_this_data_point[0] = ca_row * grid.deltax;
            // Y Coordinate
            expected_values_this_data_point[1] = (ca_col + 3 * id) * grid.deltax;
            // Z Coordinate
            expected_values_this_data_point[2] = grid.deltax * layercounter;
            int x_int = expected_values_this_data_point[0] / grid.deltax;
            int y_int = expected_values_this_data_point[1] / grid.deltax;
            // Melting time
            expected_values_this_data_point[3] = x_int * y_int;
            // Liquidus time
            expected_values_this_data_point[4] = x_int * y_int + x_int;
            // Cooling rate
            expected_values_this_data_point[5] = x_int * y_int + y_int;
            for (int nn = 0; nn < 6; nn++) {
                EXPECT_DOUBLE_EQ(expected_values_this_data_point[nn],
                                 temperature.raw_temperature_data(number_of_cells_per_rank * layercounter + n, nn));
            }
        }
    }
}

// Test unidirectional solidification problem for either directional solidification or growth of a single grain seed,
// with thermal gradient G in the domain and initial undercooling init_undercooling at the solidification interface
void testInit_UnidirectionalGradient(const std::string simulation_type, const double G,
                                     const double init_undercooling) {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Default grid struct with manually set values for test
    Grid grid;
    grid.nx = 2;
    grid.ny_local = 5;
    grid.nz = 6; // (Front is at Z = 0 for directional growth, single grain seed at Z = 2 for singlegrain problem)
    grid.domain_size = grid.nx * grid.ny_local * grid.nz;
    grid.domain_size_all_layers = grid.domain_size;
    grid.number_of_layers = 1;
    grid.layer_range = std::make_pair(0, grid.domain_size);
    int coord_z_center = Kokkos::floorf(static_cast<float>(grid.nz) / 2.0);

    // default inputs struct - manually set non-default substrateInputs values
    Inputs inputs;
    inputs.simulation_type = simulation_type;
    inputs.temperature.G = G;
    inputs.temperature.init_undercooling = init_undercooling;

    // For problems with non-zero thermal gradient, 1 K difference between each cell and its neighbor in Z
    if (G == 0)
        grid.deltax = 1 * pow(10, -6);
    else
        grid.deltax = 1.0 / G;
    inputs.domain.deltax = grid.deltax;
    double G_norm = G * grid.deltax;
    // Cells cool at rate of 1 K per time step
    inputs.temperature.R = 1000000;
    double deltat = 1 * pow(10, -6);
    inputs.domain.deltat = deltat;
    double R_norm = inputs.temperature.R * deltat;

    // Temperature struct
    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    // Test constructor initialization of _inputs
    // These should've been initialized with default values
    EXPECT_FALSE(temperature._inputs.layerwise_temp_read);
    EXPECT_EQ(temperature._inputs.temp_files_in_series, 0);
    // These should have assigned values
    EXPECT_DOUBLE_EQ(temperature._inputs.R, inputs.temperature.R);
    EXPECT_DOUBLE_EQ(temperature._inputs.G, G);
    EXPECT_DOUBLE_EQ(temperature._inputs.init_undercooling, inputs.temperature.init_undercooling);
    temperature.initialize(id, simulation_type, grid, deltat);

    // Copy temperature views back to host
    auto liquidus_cell_list_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.liquidus_cell_list);
    auto cooling_rate_list_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.cooling_rate_list);
    auto last_time_below_liquidus_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.last_time_below_liquidus);
    auto current_cooling_rate_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.current_cooling_rate);

    // Check results
    int location_init_undercooling, location_liquidus_isotherm;
    if (simulation_type == "Directional")
        location_init_undercooling = 0;
    else
        location_init_undercooling = coord_z_center;

    // If thermal gradient is 0, liquidus isotherm does not exist - initialize to nz to avoid divide by zero error and
    // ensure all cells are initialized as undercooled (i.e., at Z coordinates less than nz)
    if (G == 0)
        location_liquidus_isotherm = grid.nz;
    else
        location_liquidus_isotherm =
            location_init_undercooling + Kokkos::round(inputs.temperature.init_undercooling / (G * grid.deltax));

    // Each cell melts/solidifies once
    EXPECT_EQ(temperature.num_liquidus_times_this_layer, 2 * grid.domain_size);
    // The counter is at num_liquidus_times_this_layer since melting/resolidification aren't considered for this problem
    // type
    EXPECT_EQ(temperature.num_liquidus_times_this_layer, temperature.liquidus_time_counter);
    std::vector<int> found_liq_events_per_cell(grid.domain_size, 0);
    for (int n = 0; n < temperature.num_liquidus_times_this_layer; n++) {
        // Each cell should appear on this list twice, once for melting (time step 0, cooling rate -1) and once for
        // cooling below the liquidus (cooling rate = 1, liquidus time either 0 if G = 0 or a Z dependent value
        const int index = liquidus_cell_list_host(n);
        found_liq_events_per_cell[index]++;
        if (cooling_rate_list_host[n] == -1.0) {
            EXPECT_EQ(temperature.liquidus_time_list_host[n], 0);
        }
        else {
            EXPECT_FLOAT_EQ(cooling_rate_list_host[n], 1.0);
            const int coord_z = grid.getCoordZ(index);
            const int expected_liq_time = Kokkos::round((coord_z - location_liquidus_isotherm) * G_norm / R_norm);
            EXPECT_EQ(temperature.liquidus_time_list_host(n), expected_liq_time);
        }
    }
    // Check correct initialization of current cooling rate and last time below liquidus, which are the views used for
    // this problem type to update the temperature field
    for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
        for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
            for (int coord_y = 0; coord_y < grid.ny_local; coord_y++) {
                int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                const int expected_liq_time = Kokkos::round((coord_z - location_liquidus_isotherm) * G_norm / R_norm);
                EXPECT_EQ(last_time_below_liquidus_host(index), expected_liq_time);
                EXPECT_FLOAT_EQ(current_cooling_rate_host(index), R_norm);
                EXPECT_EQ(temperature.number_of_solidification_events(index), 1);
                EXPECT_EQ(found_liq_events_per_cell[index], 2);
            }
        }
    }
}

// Check temperature initialization for translated/mirrored FromFile or FromFinch data
template <typename MemorySpace>
void checkTemperatureResults(Inputs &inputs, Grid &grid, Temperature<MemorySpace> &temperature,
                             const int number_of_copies, const int, const int np, const bool mirror_x,
                             const int num_layers_with_points) {

    // Should have equal numbers of points associated with each layer - if initializing only 1 layer at a time, the data
    // from the other layers should not have been stored
    const int expected_num_data_points = grid.nx * number_of_copies * np * num_layers_with_points;
    EXPECT_EQ(temperature.raw_temperature_data.extent(0), expected_num_data_points);
    std::vector<int> expected_num_points_per_cell(grid.domain_size, 0);
    for (int layernumber = 0; layernumber < num_layers_with_points; layernumber++) {
        for (int n = 0; n < grid.nx * number_of_copies * np; n++) {
            int point_num = layernumber * grid.nx * number_of_copies * np + n;
            // X,Y,Z the cell would be placed at
            const int coord_x = temperature.getTempCoordX(point_num, grid.x_min, grid.deltax);
            const int coord_y = temperature.getTempCoordY(point_num, grid.y_min, grid.deltax, grid.y_offset);
            const int coord_z =
                temperature.getTempCoordZ(point_num, grid.deltax, grid.layer_height, layernumber, grid.z_min_layer);
            const int index = grid.get1DIndex(coord_x, coord_y, coord_z);
            expected_num_points_per_cell[index]++;
            // Make sure the cell is associated with the correct temperature data from "input_temperature_data"
            // TM for even numbered translations (or when mirror_x is false) is equal to the X coordinate plus any
            // applied temporal offset in Y related to its Y coordinate. If mirrored, odd numbered translations will use
            // (nx - coord_x) in place of coord_x in these calculations. TL should be (1 * layernumber) larger than TM,
            // and the cooling rate is proportional to 100 times the MPI rank the point was initalized on (in turn equal
            // to its Z coordinate)
            const bool x_coord_mirrored = ((mirror_x) && (coord_y % 2 == 1));
            double loc_along_scan;
            if (x_coord_mirrored)
                loc_along_scan = static_cast<double>((grid.nx - 1) - coord_x);
            else
                loc_along_scan = static_cast<double>(coord_x);
            const double expected_tm =
                loc_along_scan + static_cast<double>(coord_y) * inputs.temperature.temporal_offset;
            const double expected_tl = expected_tm + 1.0 * (layernumber + 1);
            const double expected_cr = 100.0 * static_cast<double>(coord_z + 1);
            EXPECT_DOUBLE_EQ(expected_tm, temperature.raw_temperature_data(point_num, 3));
            EXPECT_DOUBLE_EQ(expected_tl, temperature.raw_temperature_data(point_num, 4));
            EXPECT_DOUBLE_EQ(expected_cr, temperature.raw_temperature_data(point_num, 5));
        }
    }

    // Make sure one data point was present in each expected cell per each layer of data stored. With number_of_copies =
    // 1, data will only be present at Y = 0 locally, and each copy is translated 1 cell in Y
    for (int index = 0; index < grid.domain_size; index++) {
        const int coord_y = grid.getCoordY(index);
        const int coord_z = grid.getCoordZ(index);
        if ((coord_y < number_of_copies) && (coord_z < np)) {
            EXPECT_EQ(expected_num_points_per_cell[index], num_layers_with_points);
        }
    }
}

// Initialize test input values for InitTemperatureFromFinch and InitTemperatureFromFile
void initTestInputs(const bool mirror_x, const int number_of_copies, Inputs &inputs) {
    inputs.temperature.x_offset = 0.0;
    inputs.temperature.y_offset = 1.0 * pow(10, -6);
    inputs.temperature.mirror_x = mirror_x;
    inputs.temperature.mirror_y = false;
    inputs.temperature.temporal_offset = 100.0;
    inputs.temperature.number_of_copies = number_of_copies;
}

// Initialize test grid for InitTemperatureFromFinch and InitTemperatureFromFile
void initTestGrid(const int id, const int np, Grid &grid) {
    grid.deltax = 1.0 * pow(10, -6);
    grid.x_min = -2.0 * pow(10, -6);
    grid.nx = 2;
    grid.x_max = grid.x_min + grid.deltax * (grid.nx - 1);
    grid.y_min = 0.0;
    grid.ny_local = 3;
    grid.ny = grid.ny_local * np;
    grid.y_offset = id * grid.ny_local;
    grid.y_max = grid.y_min + grid.deltax * (grid.ny - 1);
    grid.z_min = 0.0;
    grid.z_min_layer[0] = grid.z_min;
    grid.nz = np;
    grid.domain_size = grid.nx * grid.ny_local * grid.nz;
    grid.domain_size_all_layers = grid.domain_size;
    grid.number_of_layers = 1;
    grid.layer_height = 0;
    grid.layer_range = std::make_pair(0, grid.domain_size);
}

// Test storing temperature data from Finch on the correct ExaCA ranks, optionally mirroring and translating the data
void testInitTemperatureFromFinch(const bool mirror_x, const int number_of_copies, const bool layerwise_temp_init,
                                  const int number_of_layers) {

    using memory_space = TEST_MEMSPACE;
    using view_type_coupled = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Default inputs struct - manually set temperature values for test
    Inputs inputs;
    inputs.temperature.layerwise_temp_read = layerwise_temp_init;
    inputs.temperature.temp_files_in_series = number_of_layers;
    // Default grid struct - manually set up domain for test
    Grid grid;
    initTestInputs(mirror_x, number_of_copies, inputs);
    initTestGrid(id, np, grid);
    grid.number_of_layers = number_of_layers;

    // Create dummy Finch temperature data stored on various ranks, to be mapped to ExaCA ranks. Each layer has nx * np
    // locations with data
    const int locations_with_data = np * grid.nx * number_of_layers;
    int num_layers_with_points;
    if (layerwise_temp_init)
        num_layers_with_points = 1;
    else
        num_layers_with_points = number_of_layers;
    view_type_coupled input_temperature_data(Kokkos::ViewAllocateWithoutInitializing("FinchData"), locations_with_data,
                                             6);
    std::vector<int> first_value, last_value;
    int finch_counter = 0;
    for (int layernumber = 0; layernumber < num_layers_with_points; layernumber++) {
        first_value.push_back(finch_counter);
        // Each rank has Finch data at a given Z location, at all relevant X values and at every third Y coordinate (so
        // each ExaCA rank after rearranging the data will only own data from one of the Y coordinates). Number of data
        // points proportional to the number of MPI ranks and varies based on the layer number. Liquidus times are
        // offset from the melting times by a value proportional to the layer number
        for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
            for (int coord_y = 0; coord_y < np; coord_y++) {
                input_temperature_data(finch_counter, 0) = grid.x_min + grid.deltax * coord_x;
                input_temperature_data(finch_counter, 1) = grid.y_min + 3.0 * coord_y * grid.deltax;
                input_temperature_data(finch_counter, 2) = grid.z_min + id * grid.deltax;
                input_temperature_data(finch_counter, 3) = static_cast<double>(coord_x);
                input_temperature_data(finch_counter, 4) = static_cast<double>(coord_x) + 1.0 * (layernumber + 1);
                input_temperature_data(finch_counter, 5) = 100.0 * (id + 1);
                finch_counter++;
            }
        }
        last_value.push_back(finch_counter);
    }

    // Construct temperature views
    Temperature<memory_space> temperature(id, np, grid, inputs.temperature, inputs.print, input_temperature_data, first_value,
                                          last_value);

    // Check that the right ranks have the right temperature data
    checkTemperatureResults(inputs, grid, temperature, number_of_copies, id, np, mirror_x, num_layers_with_points);
}

// Test storing temperature data from a file on the correct ExaCA ranks, using the same process at the FromFinch test
void testInitTemperatureFromFile(const bool mirror_x, const int number_of_copies) {

    using memory_space = TEST_MEMSPACE;

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Default inputs struct - manually set temperature values for test
    Inputs inputs;
    // Default grid struct - manually set up domain for test
    Grid grid;
    initTestInputs(mirror_x, number_of_copies, inputs);
    initTestGrid(id, np, grid);

    // Create dummy file temperature data on rank 0, including data to be mapped to other ExaCA ranks
    if (id == 0) {
        std::ofstream test_data_file;
        test_data_file.open("TestFromFile.csv");
        test_data_file << "x, y, z, tm, tl, cr" << std::endl;
        for (int coord_z = 0; coord_z < np; coord_z++) {
            for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
                for (int coord_y = 0; coord_y < np; coord_y++) {
                    test_data_file << grid.x_min + grid.deltax * coord_x << ","
                                   << grid.y_min + 3.0 * coord_y * grid.deltax << ","
                                   << grid.z_min + coord_z * grid.deltax << "," << static_cast<double>(coord_x) << ","
                                   << static_cast<double>(coord_x) + 1.0 << ","
                                   << 100.0 * static_cast<double>(coord_z + 1) << std::endl;
                }
            }
        }
        test_data_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Construct temperature views and read data back from files
    inputs.temperature.temp_paths.push_back("TestFromFile.csv");
    inputs.temperature.temp_files_in_series = 1;
    inputs.temperature.layerwise_temp_read = false;

    Temperature<memory_space> temperature(grid, inputs.temperature, inputs.print);
    temperature.readTemperatureData(id, grid, 0);

    // Check that the right ranks have the right temperature data
    checkTemperatureResults(inputs, grid, temperature, number_of_copies, id, np, mirror_x, grid.number_of_layers);
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, temperature) {
    // Multiple permutations of inputs: number_of_layers, layerwise_temp_read, test_binary_input_read
    // reading temperature data is performed in the same manner with and without remelting
    std::vector<int> number_of_layers_vals = {1, 2, 2, 2};
    std::vector<bool> layerwise_temp_read_vals = {false, true, false, true};
    std::vector<bool> test_binary_input_read_vals = {false, false, false, true};
    int num_vals = test_binary_input_read_vals.size();
    for (int test_count = 0; test_count < num_vals; test_count++) {
        testReadTemperatureData(number_of_layers_vals[test_count], layerwise_temp_read_vals[test_count],
                                test_binary_input_read_vals[test_count]);
    }
    // Test for directional and single grain problems, and with and without a thermal gradient for the single grain
    // problem and with/without an initial undercooling present at the initial solidification front
    testInit_UnidirectionalGradient("Directional", 1000000, 0);
    testInit_UnidirectionalGradient("Directional", 1000000, 2);
    testInit_UnidirectionalGradient("SingleGrain", 0, 2);
    testInit_UnidirectionalGradient("SingleGrain", 1000000, 2);
    // Coupled Finch-ExaCA simulation init tests: mirror_x (true/false), number_of_copies, layerwise_temp_init
    // (true/false), number_of_layers
    std::vector<bool> mirror_x_vals = {false, false, true, false, false};
    std::vector<int> number_of_copies_vals = {1, 3, 3, 1, 1};
    std::vector<bool> layerwise_temp_init_vals = {false, false, false, false, true};
    std::vector<int> number_of_layers_finch_vals = {1, 1, 1, 2, 2};
    num_vals = mirror_x_vals.size();
    for (int test_count = 0; test_count < num_vals; test_count++) {
        testInitTemperatureFromFinch(mirror_x_vals[test_count], number_of_copies_vals[test_count],
                                     layerwise_temp_init_vals[test_count], number_of_layers_finch_vals[test_count]);
    }
    testInitTemperatureFromFile(false, 1);
    testInitTemperatureFromFile(false, 3);
    testInitTemperatureFromFile(true, 3);
}
} // end namespace Test
