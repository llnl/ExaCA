// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAinputs.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// file_read_tests
//---------------------------------------------------------------------------//
void writeTestData(std::string input_filename, int print_version) {

    std::ofstream test_data_file;
    test_data_file.open(input_filename);
    // Write required inputs to all files
    test_data_file << "{" << std::endl;
    test_data_file << "   \"SimulationType\": \"FromFile\"," << std::endl;
    test_data_file << "   \"MaterialFileName\": \"Inconel625.json\"," << std::endl;
    test_data_file << "   \"CrystalOrientationFile\": \"CrystalOrientationVectors.csv\"," << std::endl;
    test_data_file << "   \"RandomSeed\": 2," << std::endl;
    test_data_file << "   \"Domain\": {" << std::endl;
    test_data_file << "      \"CellSize\": 1," << std::endl;
    test_data_file << "      \"TimeStep\": 1.5," << std::endl;
    test_data_file << "      \"NumberOfLayers\": 2," << std::endl;
    test_data_file << "      \"LayerOffset\": 1" << std::endl;
    test_data_file << "   }," << std::endl;
    test_data_file << "   \"Nucleation\": {" << std::endl;
    test_data_file << "      \"Density\": 10," << std::endl;
    test_data_file << "      \"MeanUndercooling\": 5," << std::endl;
    test_data_file << "      \"StDev\": 0.5" << std::endl;
    test_data_file << "   }," << std::endl;
    test_data_file << "   \"TemperatureData\": {" << std::endl;
    test_data_file << "      \"TemperatureFiles\": [\".//1DummyTemperature.txt\",\".//2DummyTemperature.txt\"]"
                   << std::endl;
    test_data_file << "   }," << std::endl;
    test_data_file << "   \"Substrate\": {" << std::endl;
    // FIXME: old input PowderDensity used here to check that the expected mean powder grain spacing ends up correct,
    // change this to MeanPowderGrainSize when deprecated input compatibility is removed
    test_data_file << "      \"SubstrateFilename\": \"DummySubstrate.vtk\"," << std::endl;
    test_data_file << "      \"PowderDensity\": 1000," << std::endl;
    test_data_file << "      \"BaseplateTopZ\": -0.00625" << std::endl;
    test_data_file << "   }," << std::endl;
    test_data_file << "   \"Printing\": {" << std::endl;
    test_data_file << "      \"PathToOutput\": \"ExaCA\"," << std::endl;
    test_data_file << "      \"OutputFile\": \"Test\"," << std::endl;
    test_data_file << "      \"PrintBinary\": true," << std::endl;
    // two different permutations of print outputs - deprecated and non-deprecated GrainMisorientation
    if (print_version == 0) {
        test_data_file << "      \"PrintExaConstitSize\": 0," << std::endl;
        test_data_file << "      \"Intralayer\": {" << std::endl;
        test_data_file << "          \"Fields\": "
                          "[\"UndercoolingChange\",\"GrainID\",\"LayerID\",\"MeltTimeStep\",\"CritTimeStep\"],"
                       << std::endl;
        test_data_file << "          \"Increment\": 200," << std::endl;
        test_data_file << "          \"PrintIdleFrames\": true" << std::endl;
        test_data_file << "      }," << std::endl;
        test_data_file << "      \"Interlayer\": {" << std::endl;
        test_data_file
            << "          \"Fields\": [\"UndercoolingCurrent\",\"GrainMisorientation\",\"GrainID\",\"LayerID\"]"
            << std::endl;
        test_data_file << "      }" << std::endl;
    }
    else if (print_version == 1) {
        test_data_file << "      \"PrintExaConstitSize\": 500," << std::endl;
        test_data_file << "      \"Interlayer\": {" << std::endl;
        test_data_file
            << "          \"Fields\": [\"CrystalMisorientationX\",\"CrystalMisorientationY\",\"GrainID\",\"LayerID\"]"
            << std::endl;
        test_data_file << "      }" << std::endl;
    }
    test_data_file << "   }" << std::endl;
    test_data_file << "}" << std::endl;
    test_data_file.close();
}

void testInputs(int print_version) {

    int id, np;
    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // DirSolidification, SpotMelt, TwoGrainDirSolidification were installed from the examples directory
    // Since no temperature files exist in the repo, and there is no ability to write temperature files to a different
    // directory ( would need examples/Temperatures) using the C++11 standard, dummy input files are written and parsed
    // to test an example problem that uses temperature data from a file.
    std::vector<std::string> input_filenames = {"Inp_DirSolidification", "Inp_SpotMelt", "Inp_TemperatureTest",
                                                "Inp_TwoGrainDirSolidification"};
    for (int n = 0; n < 4; n++) {
        input_filenames[n] += ".json";
    }
    std::vector<std::string> temperature_fnames = {"1DummyTemperature.txt", "2DummyTemperature.txt"};

    // On rank 0, write dummy input files for using read temperature data (input_filenames[2] and [3])
    if (id == 0) {
        writeTestData(input_filenames[2], print_version);

        // Create test temperature files "1DummyTemperature.txt", "2DummyTemperature.txt"
        std::ofstream test_temp_1, test_temp_2;
        test_temp_1.open(temperature_fnames[0]);
        test_temp_2.open(temperature_fnames[1]);
        test_temp_1 << "X" << std::endl;
        test_temp_2 << "X" << std::endl;
        test_temp_1.close();
        test_temp_2.close();

        // Create test substrate file "DummySubstrate.txt"
        std::ofstream test_sub;
        test_sub.open("DummySubstrate.txt");
        test_sub << "X" << std::endl;
        test_sub.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Read and parse each input file
    for (auto filename : input_filenames) {
        std::cout << "Reading " << filename << std::endl;
        // Data printing structure - contains print options (false by default) and functions
        Inputs inputs(id, filename);
        MPI_Barrier(MPI_COMM_WORLD);

        // Check the results
        // The existence of the specified orientation, substrate, and temperature filenames was already checked within
        // inputs constructor
        // These should be the same for all test problems (except the 4th one, which has 0 nucleation density)
        EXPECT_DOUBLE_EQ(inputs.domain.deltax, 1.0 * pow(10, -6));
        if (filename == input_filenames[3])
            EXPECT_DOUBLE_EQ(inputs.nucleation.n_max, 0.0);
        else
            EXPECT_DOUBLE_EQ(inputs.nucleation.n_max, 1.0 * pow(10, 13));
        EXPECT_DOUBLE_EQ(inputs.nucleation.dtn, 5.0);
        EXPECT_DOUBLE_EQ(inputs.nucleation.dtsigma, 0.5);
        // Raw values read from material input file - not normalized (normalization occurs on local copies of these in
        // the IRF constructor)
        EXPECT_FLOAT_EQ(inputs.irf.A[0], -0.00000010302);
        EXPECT_FLOAT_EQ(inputs.irf.B[0], 0.00010533);
        EXPECT_FLOAT_EQ(inputs.irf.C[0], 0.0022196);
        EXPECT_FLOAT_EQ(inputs.irf.D[0], 0);
        EXPECT_FLOAT_EQ(inputs.irf.freezing_range[0], 210);

        // These are different for all 3 test problems
        if ((filename == input_filenames[0]) || (filename == input_filenames[3])) {
            EXPECT_DOUBLE_EQ(inputs.temperature.G, 500000.0);
            EXPECT_DOUBLE_EQ(inputs.temperature.R, 300000.0);
            // compare with float to avoid floating point error with irrational number
            float deltat_comp = static_cast<float>(inputs.domain.deltat);
            EXPECT_FLOAT_EQ(deltat_comp, pow(10, -6) / 15.0);
            EXPECT_EQ(inputs.domain.nx, 200);
            EXPECT_EQ(inputs.domain.ny, 200);
            EXPECT_EQ(inputs.domain.nz, 200);
            if (filename == input_filenames[0]) {
                EXPECT_DOUBLE_EQ(inputs.substrate.fract_surface_sites_active, 0.08);
                EXPECT_TRUE(inputs.print.base_filename == "TestProblemDirS");
                EXPECT_TRUE(inputs.substrate.surface_init_mode == "SurfaceSiteFraction");
                EXPECT_DOUBLE_EQ(inputs.temperature.init_undercooling, 0.0);
            }
            else {
                EXPECT_EQ(inputs.substrate.grain_locations_x[0], 100);
                EXPECT_EQ(inputs.substrate.grain_locations_y[0], 50);
                EXPECT_EQ(inputs.substrate.grain_ids[0], 25);
                EXPECT_EQ(inputs.substrate.grain_locations_x[1], 100);
                EXPECT_EQ(inputs.substrate.grain_locations_y[1], 150);
                EXPECT_EQ(inputs.substrate.grain_ids[1], 9936);
                EXPECT_TRUE(inputs.print.base_filename == "TestProblemTwoGrainDirS");
                EXPECT_TRUE(inputs.substrate.surface_init_mode == "Custom");
                EXPECT_DOUBLE_EQ(inputs.temperature.init_undercooling, 10.0);
            }
            EXPECT_TRUE(inputs.print.intralayer);
            EXPECT_EQ(inputs.print.intralayer_increment, 5250);
            EXPECT_FALSE(inputs.print.intralayer_idle_frames);
            EXPECT_FALSE(inputs.print.intralayer_grain_id);
            EXPECT_FALSE(inputs.print.intralayer_layer_id);
            EXPECT_FALSE(inputs.print.intralayer_crystal_misorientation[0]);
            EXPECT_FALSE(inputs.print.intralayer_crystal_misorientation[1]);
            EXPECT_TRUE(inputs.print.intralayer_crystal_misorientation[2]);
            EXPECT_FALSE(inputs.print.intralayer_undercooling_current);
            EXPECT_FALSE(inputs.print.intralayer_melt_time_step);
            EXPECT_FALSE(inputs.print.intralayer_crit_time_step);
            EXPECT_FALSE(inputs.print.intralayer_undercooling_change);
            EXPECT_FALSE(inputs.print.intralayer_cell_type);
            EXPECT_FALSE(inputs.print.intralayer_diagonal_length);
            EXPECT_FALSE(inputs.print.intralayer_solidification_event_counter);
            EXPECT_FALSE(inputs.print.intralayer_number_of_solidification_events);
            EXPECT_TRUE(inputs.print.interlayer_full);
            EXPECT_FALSE(inputs.print.interlayer_current);
            EXPECT_TRUE(inputs.print.interlayer_grain_id);
            EXPECT_TRUE(inputs.print.interlayer_layer_id);
            EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[0]);
            EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[1]);
            EXPECT_TRUE(inputs.print.interlayer_crystal_misorientation[2]);
            EXPECT_FALSE(inputs.print.interlayer_undercooling_current);
            EXPECT_FALSE(inputs.print.interlayer_melt_time_step);
            EXPECT_FALSE(inputs.print.interlayer_crit_time_step);
            EXPECT_FALSE(inputs.print.interlayer_undercooling_change);
            EXPECT_FALSE(inputs.print.interlayer_cell_type);
            EXPECT_FALSE(inputs.print.interlayer_diagonal_length);
            EXPECT_FALSE(inputs.print.interlayer_solidification_event_counter);
            EXPECT_FALSE(inputs.print.interlayer_number_of_solidification_events);
            EXPECT_EQ(inputs.print.print_layer_number[0], 0);
        }
        else if (filename == input_filenames[1]) {
            EXPECT_DOUBLE_EQ(inputs.temperature.G, 500000.0);
            EXPECT_DOUBLE_EQ(inputs.temperature.R, 300000.0);
            // compare with float to avoid floating point error with irrational number
            float deltat_comp = static_cast<float>(inputs.domain.deltat);
            EXPECT_FLOAT_EQ(deltat_comp, pow(10, -6) / 15.0);
            EXPECT_EQ(inputs.domain.spot_radius, 75);
            EXPECT_FALSE(inputs.substrate.use_substrate_file);
            EXPECT_FALSE(inputs.substrate.baseplate_through_powder);
            EXPECT_FLOAT_EQ(inputs.substrate.baseplate_grain_spacing, 25.0);
            EXPECT_TRUE(inputs.print.base_filename == "TestProblemSpot");
            EXPECT_TRUE(inputs.print.intralayer);
            EXPECT_EQ(inputs.print.intralayer_increment, 2000);
            EXPECT_TRUE(inputs.print.intralayer_idle_frames);
            EXPECT_FALSE(inputs.print.intralayer_grain_id);
            EXPECT_FALSE(inputs.print.intralayer_layer_id);
            EXPECT_FALSE(inputs.print.intralayer_crystal_misorientation[0]);
            EXPECT_FALSE(inputs.print.intralayer_crystal_misorientation[1]);
            EXPECT_TRUE(inputs.print.intralayer_crystal_misorientation[2]);
            EXPECT_FALSE(inputs.print.intralayer_undercooling_current);
            EXPECT_TRUE(inputs.print.intralayer_melt_time_step);
            EXPECT_TRUE(inputs.print.intralayer_crit_time_step);
            EXPECT_FALSE(inputs.print.intralayer_undercooling_change);
            EXPECT_FALSE(inputs.print.intralayer_cell_type);
            EXPECT_FALSE(inputs.print.intralayer_diagonal_length);
            EXPECT_FALSE(inputs.print.intralayer_solidification_event_counter);
            EXPECT_FALSE(inputs.print.intralayer_number_of_solidification_events);
            EXPECT_TRUE(inputs.print.interlayer_full);
            EXPECT_FALSE(inputs.print.interlayer_current);
            EXPECT_TRUE(inputs.print.interlayer_grain_id);
            EXPECT_TRUE(inputs.print.interlayer_layer_id);
            EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[0]);
            EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[1]);
            EXPECT_TRUE(inputs.print.interlayer_crystal_misorientation[2]);
            EXPECT_FALSE(inputs.print.interlayer_undercooling_current);
            EXPECT_FALSE(inputs.print.interlayer_melt_time_step);
            EXPECT_FALSE(inputs.print.interlayer_crit_time_step);
            EXPECT_FALSE(inputs.print.interlayer_undercooling_change);
            EXPECT_FALSE(inputs.print.interlayer_cell_type);
            EXPECT_FALSE(inputs.print.interlayer_diagonal_length);
            EXPECT_FALSE(inputs.print.interlayer_solidification_event_counter);
            EXPECT_FALSE(inputs.print.interlayer_number_of_solidification_events);
            EXPECT_EQ(inputs.print.print_layer_number[0], 0);
            EXPECT_FALSE(inputs.print.print_default_rve);
            EXPECT_DOUBLE_EQ(inputs.rng_seed, 0.0);
        }
        else if (filename == input_filenames[2]) {
            EXPECT_DOUBLE_EQ(inputs.domain.deltat, 1.5 * pow(10, -6));
            EXPECT_EQ(inputs.temperature.temp_files_in_series, 2);
            EXPECT_EQ(inputs.domain.number_of_layers, 2);
            EXPECT_EQ(inputs.domain.layer_height, 1);
            EXPECT_TRUE(inputs.substrate.use_substrate_file);
            EXPECT_FALSE(inputs.temperature.layerwise_temp_read);
            EXPECT_DOUBLE_EQ(inputs.substrate.powder_grain_spacing, 10.0);
            // -0.00625 was input
            EXPECT_DOUBLE_EQ(inputs.substrate.baseplate_top_z, -0.00625);
            EXPECT_TRUE(inputs.print.base_filename == "Test");
            EXPECT_TRUE(inputs.temperature.temp_paths[0] == ".//1DummyTemperature.txt");
            EXPECT_TRUE(inputs.temperature.temp_paths[1] == ".//2DummyTemperature.txt");
            if (print_version == 0) {
                EXPECT_TRUE(inputs.print.intralayer);
                EXPECT_EQ(inputs.print.intralayer_increment, 200);
                EXPECT_TRUE(inputs.print.intralayer_idle_frames);
                EXPECT_TRUE(inputs.print.intralayer_grain_id);
                EXPECT_TRUE(inputs.print.intralayer_layer_id);
                for (int dir = 0; dir < 2; dir++)
                    EXPECT_FALSE(inputs.print.intralayer_crystal_misorientation[dir]);
                EXPECT_FALSE(inputs.print.intralayer_undercooling_current);
                EXPECT_TRUE(inputs.print.intralayer_melt_time_step);
                EXPECT_TRUE(inputs.print.intralayer_crit_time_step);
                EXPECT_TRUE(inputs.print.intralayer_undercooling_change);
                EXPECT_FALSE(inputs.print.intralayer_cell_type);
                EXPECT_FALSE(inputs.print.intralayer_diagonal_length);
                EXPECT_FALSE(inputs.print.intralayer_solidification_event_counter);
                EXPECT_FALSE(inputs.print.intralayer_number_of_solidification_events);
                EXPECT_TRUE(inputs.print.interlayer_full);
                EXPECT_FALSE(inputs.print.interlayer_current);
                EXPECT_TRUE(inputs.print.interlayer_grain_id);
                EXPECT_TRUE(inputs.print.interlayer_layer_id);
                EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[0]);
                EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[1]);
                EXPECT_TRUE(inputs.print.interlayer_crystal_misorientation[2]);
                EXPECT_TRUE(inputs.print.interlayer_undercooling_current);
                EXPECT_FALSE(inputs.print.interlayer_melt_time_step);
                EXPECT_FALSE(inputs.print.interlayer_crit_time_step);
                EXPECT_FALSE(inputs.print.interlayer_undercooling_change);
                EXPECT_FALSE(inputs.print.interlayer_cell_type);
                EXPECT_FALSE(inputs.print.interlayer_diagonal_length);
                EXPECT_FALSE(inputs.print.interlayer_solidification_event_counter);
                EXPECT_FALSE(inputs.print.interlayer_number_of_solidification_events);
                EXPECT_EQ(inputs.print.print_layer_number[0], 1);
                EXPECT_FALSE(inputs.print.print_default_rve);
            }
            else if (print_version == 1) {
                EXPECT_FALSE(inputs.print.intralayer);
                EXPECT_TRUE(inputs.print.interlayer_full);
                EXPECT_FALSE(inputs.print.interlayer_current);
                EXPECT_TRUE(inputs.print.interlayer_grain_id);
                EXPECT_TRUE(inputs.print.interlayer_layer_id);
                EXPECT_TRUE(inputs.print.interlayer_crystal_misorientation[0]);
                EXPECT_TRUE(inputs.print.interlayer_crystal_misorientation[1]);
                EXPECT_FALSE(inputs.print.interlayer_crystal_misorientation[2]);
                EXPECT_FALSE(inputs.print.interlayer_undercooling_current);
                EXPECT_FALSE(inputs.print.interlayer_melt_time_step);
                EXPECT_FALSE(inputs.print.interlayer_crit_time_step);
                EXPECT_FALSE(inputs.print.interlayer_undercooling_change);
                EXPECT_FALSE(inputs.print.interlayer_cell_type);
                EXPECT_FALSE(inputs.print.interlayer_diagonal_length);
                EXPECT_FALSE(inputs.print.interlayer_solidification_event_counter);
                EXPECT_FALSE(inputs.print.interlayer_number_of_solidification_events);
                EXPECT_EQ(inputs.print.print_layer_number[0], 1);
                EXPECT_TRUE(inputs.print.print_default_rve);
            }
            EXPECT_FALSE(inputs.temperature.layerwise_temp_read);
            EXPECT_DOUBLE_EQ(inputs.rng_seed, 2.0);
            EXPECT_TRUE(inputs.print.print_binary);
        }

        // Empty grid and timer structs for log file print, give dummy values
        Grid grid;
        grid.nx = 0;
        grid.ny = 0;
        grid.nz = 0;
        grid.deltax = 0.0;
        grid.x_min = 0.0;
        grid.y_min = 0.0;
        grid.z_min = 0.0;
        grid.x_max = 0.0;
        grid.y_max = 0.0;
        grid.z_max = 0.0;
        Timers timers(0);

        // Print log file
        inputs.printExaCALog(0, 1, 0, grid, timers, 0.0);

        // Check that log file can be parsed with json
        std::string log_filename = inputs.print.path_to_output + inputs.print.base_filename + ".json";
        std::ifstream input_data_stream(log_filename);
        nlohmann::json test_input_data = nlohmann::json::parse(input_data_stream);
    }
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, inputs) {
    // input argument: test for two different print versions for last input file
    testInputs(0);
    testInputs(1);
}
} // end namespace Test
