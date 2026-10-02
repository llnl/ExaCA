// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "GArepresentativeregion.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// Tests of representative region structure
//---------------------------------------------------------------------------//

void writeTestArea() {
    // Write out example data
    std::ofstream test_data;
    test_data.open("TestArea.json");
    test_data << "{" << std::endl;
    test_data << "   \"Regions\": {" << std::endl;
    test_data << "       \"RepresentativeArea\": {" << std::endl;
    test_data << "          \"units\": \"Meters\"," << std::endl;
    test_data << "          \"zBounds\": [0.000005,0.000005]," << std::endl;
    test_data << "          \"printPoleFigureData\": true," << std::endl;
    test_data << "          \"printInversePoleFigureData\": true," << std::endl;
    test_data << "          \"printAvgStats\": [\"GrainTypeFractions\"]," << std::endl;
    test_data << "          \"printPerGrainStats\": [\"IPFZ-RGB\", \"Size\"]" << std::endl;
    test_data << "      }" << std::endl;
    test_data << "   }" << std::endl;
    test_data << "}" << std::endl;
    test_data.close();
}

void writeTestVolume() {
    // Write out example data
    std::ofstream test_data;
    test_data.open("TestVolume.json");
    test_data << "{" << std::endl;
    test_data << "   \"Regions\": {" << std::endl;
    test_data << "       \"RepresentativeVolume\": {" << std::endl;
    test_data << "          \"units\": \"Cells\"," << std::endl;
    test_data << "          \"xBounds\": [0, 4]," << std::endl;
    test_data << "          \"yBounds\": [1, 10]," << std::endl;
    test_data << "          \"zBounds\": [0, 9]," << std::endl;
    test_data << "          \"printExaConstit\": false," << std::endl;
    test_data << "          \"printPoleFigureData\": true," << std::endl;
    test_data << "          \"printAvgStats\": [\"GrainTypeFractions\", \"Misorientation\", \"Size\", "
                 "\"BuildTransAspectRatio\", \"XExtent\", \"YExtent\", \"ZExtent\"],"
              << std::endl;
    test_data << "          \"printPerGrainStats\": [\"IPFZ-RGB\", \"Size\"]," << std::endl;
    test_data << "          \"printPerZCoordinateStats\": [\"MeanGrainArea\"]" << std::endl;
    test_data << "      }" << std::endl;
    test_data << "   }" << std::endl;
    test_data << "}" << std::endl;
    test_data.close();
}

void writeTestVolume_FullDomain(const int nx, const int ny, const int nz) {
    // Write out example data for checking grain extents for full domain
    std::ofstream test_data;
    test_data.open("TestVolume.json");
    test_data << "{" << std::endl;
    test_data << "   \"Regions\": {" << std::endl;
    test_data << "       \"RepresentativeVolume\": {" << std::endl;
    test_data << "          \"units\": \"Cells\"," << std::endl;
    test_data << "          \"xBounds\": [0, " << std::to_string(nx - 1) << "]," << std::endl;
    test_data << "          \"yBounds\": [0, " << std::to_string(ny - 1) << "]," << std::endl;
    test_data << "          \"zBounds\": [0, " << std::to_string(nz - 1) << "]," << std::endl;
    test_data << "          \"printExaConstit\": false," << std::endl;
    test_data << "          \"printPoleFigureData\": false," << std::endl;
    test_data << "          \"printAvgStats\": [\"XExtent\", \"YExtent\", \"ZExtent\"]" << std::endl;
    test_data << "      }" << std::endl;
    test_data << "   }" << std::endl;
    test_data << "}" << std::endl;
    test_data.close();
}

void testConstructRepresentativeRegion_Volume() {
    // Write out example data
    writeTestVolume();

    // Read in test data
    std::ifstream analysis_data_stream("TestVolume.json");
    nlohmann::json AnalysisData = nlohmann::json::parse(analysis_data_stream);
    // Domain bounds
    const int nx = 20;
    const int ny = 30;
    const int nz = 40;
    double deltax = 1.25 * Kokkos::pow(10, -6);
    double x_min = -1 * Kokkos::pow(10, -6);
    double y_min = 0;
    double z_min = 0;
    double x_max = x_min + (nx - 1) * deltax;
    double y_max = y_min + (ny - 1) * deltax;
    double z_max = z_min + (nz - 1) * deltax;
    std::vector<double> xyz_bounds = {x_min, y_min, z_min, x_max, y_max, z_max};
    // View for storing grain ID data
    Kokkos::View<int ***, Kokkos::HostSpace> grain_id(Kokkos::ViewAllocateWithoutInitializing("grain_id"), nz, nx, ny);
    Kokkos::View<short ***, Kokkos::HostSpace> phase_id(Kokkos::ViewAllocateWithoutInitializing("phase_id"), nz, nx,
                                                        ny);

    // Construct region
    RepresentativeRegion representativeregion(AnalysisData, "RepresentativeVolume", nx, ny, nz, deltax, xyz_bounds,
                                              grain_id, phase_id, false);

    // Check results
    EXPECT_TRUE(representativeregion.region_type == "volume");
    EXPECT_TRUE(representativeregion.region_orientation == "XYZ");
    EXPECT_DOUBLE_EQ(representativeregion.x_bounds_meters[0], x_min);
    EXPECT_DOUBLE_EQ(representativeregion.y_bounds_meters[0], deltax);
    EXPECT_DOUBLE_EQ(representativeregion.z_bounds_meters[0], z_min);
    EXPECT_DOUBLE_EQ(representativeregion.x_bounds_meters[1], x_min + deltax * 4);
    EXPECT_DOUBLE_EQ(representativeregion.y_bounds_meters[1], y_min + deltax * 10);
    EXPECT_DOUBLE_EQ(representativeregion.z_bounds_meters[1], z_min + deltax * 9);
    double expected_region_size_meters =
        (representativeregion.x_bounds_meters[1] - representativeregion.x_bounds_meters[0] + deltax) *
        (representativeregion.y_bounds_meters[1] - representativeregion.y_bounds_meters[0] + deltax) *
        (representativeregion.z_bounds_meters[1] - representativeregion.z_bounds_meters[0] + deltax);
    double expected_region_size_microns = expected_region_size_meters * Kokkos::pow(10, 18);
    EXPECT_DOUBLE_EQ(representativeregion.region_size_microns, expected_region_size_microns);
    EXPECT_DOUBLE_EQ(representativeregion.region_size_meters, expected_region_size_meters);
    EXPECT_EQ(representativeregion.x_bounds_cells[0], 0);
    EXPECT_EQ(representativeregion.y_bounds_cells[0], 1);
    EXPECT_EQ(representativeregion.z_bounds_cells[0], 0);
    EXPECT_EQ(representativeregion.x_bounds_cells[1], 4);
    EXPECT_EQ(representativeregion.y_bounds_cells[1], 10);
    EXPECT_EQ(representativeregion.z_bounds_cells[1], 9);
    int expected_region_size_cells = 500;
    EXPECT_EQ(representativeregion.region_size_cells, expected_region_size_cells);
    int num_analysis_options_stats = representativeregion.analysis_options_stats_yn.size();
    for (int i = 0; i < num_analysis_options_stats; i++)
        EXPECT_TRUE(representativeregion.analysis_options_stats_yn[i]);
    int num_analysis_options_per_grain_stats = representativeregion.analysis_options_per_grain_stats_yn.size();
    for (int i = 0; i < num_analysis_options_per_grain_stats; i++) {
        if ((i == 1) || (i == 6))
            EXPECT_TRUE(representativeregion.analysis_options_per_grain_stats_yn[i]);
        else
            EXPECT_FALSE(representativeregion.analysis_options_per_grain_stats_yn[i]);
    }
    int num_analysis_options_per_z_stats = representativeregion.analysis_options_per_z_stats_yn.size();
    for (int i = 0; i < num_analysis_options_per_z_stats; i++)
        EXPECT_TRUE(representativeregion.analysis_options_per_z_stats_yn[i]);
    EXPECT_TRUE(representativeregion.print_per_grain_stats_yn);
    EXPECT_TRUE(representativeregion.print_pole_figure_yn);
    EXPECT_FALSE(representativeregion.print_exaconstit_yn);
    EXPECT_FALSE(representativeregion.print_inverse_pole_figure_map_yn);
}

// Initialize a representative area from json format
void testConstructRepresentativeRegion_Area() {

    // Write out example data
    writeTestArea();

    // Read in test data
    std::ifstream analysis_data_stream("TestArea.json");
    nlohmann::json analysis_data = nlohmann::json::parse(analysis_data_stream);

    // Domain bounds
    const int nx = 10;
    const int ny = 20;
    const int nz = 20;
    double deltax = 1.25 * Kokkos::pow(10, -6);
    double x_min = 0;
    double y_min = 0;
    double z_min = 0;
    double x_max = x_min + (nx - 1) * deltax;
    double y_max = y_min + (ny - 1) * deltax;
    double z_max = z_min + (nz - 1) * deltax;
    std::vector<double> xyz_bounds = {x_min, y_min, z_min, x_max, y_max, z_max};
    // View for storing grain ID data
    Kokkos::View<int ***, Kokkos::HostSpace> grain_id(Kokkos::ViewAllocateWithoutInitializing("grain_id"), nz, nx, ny);
    Kokkos::View<short ***, Kokkos::HostSpace> phase_id(Kokkos::ViewAllocateWithoutInitializing("phase_id"), nz, nx,
                                                        ny);

    // Construct region
    RepresentativeRegion representativeregion(analysis_data, "RepresentativeArea", nx, ny, nz, deltax, xyz_bounds,
                                              grain_id, phase_id, false);

    // Check results
    EXPECT_TRUE(representativeregion.region_type == "area");
    EXPECT_TRUE(representativeregion.region_orientation == "XY");
    EXPECT_DOUBLE_EQ(representativeregion.x_bounds_meters[0], x_min);
    EXPECT_DOUBLE_EQ(representativeregion.y_bounds_meters[0], y_min);
    EXPECT_DOUBLE_EQ(representativeregion.z_bounds_meters[0], 0.000005);
    EXPECT_DOUBLE_EQ(representativeregion.x_bounds_meters[1], x_max);
    EXPECT_DOUBLE_EQ(representativeregion.y_bounds_meters[1], y_max);
    EXPECT_DOUBLE_EQ(representativeregion.z_bounds_meters[1], 0.000005);
    EXPECT_EQ(representativeregion.x_bounds_cells[0], 0);
    EXPECT_EQ(representativeregion.y_bounds_cells[0], 0);
    EXPECT_EQ(representativeregion.z_bounds_cells[0], 4);
    EXPECT_EQ(representativeregion.x_bounds_cells[1], 9);
    EXPECT_EQ(representativeregion.y_bounds_cells[1], 19);
    EXPECT_EQ(representativeregion.z_bounds_cells[1], 4);
    int expected_region_size_cells = 200;
    EXPECT_EQ(representativeregion.region_size_cells, expected_region_size_cells);
    double expected_region_size_meters = expected_region_size_cells * Kokkos::pow(deltax, 2);
    double expected_region_size_microns = expected_region_size_meters * Kokkos::pow(10, 12);
    EXPECT_DOUBLE_EQ(representativeregion.region_size_microns, expected_region_size_microns);
    EXPECT_DOUBLE_EQ(representativeregion.region_size_meters, expected_region_size_meters);
    EXPECT_TRUE(representativeregion.analysis_options_stats_yn[0]);
    int num_analysis_options_stats = representativeregion.analysis_options_stats_yn.size();
    for (int i = 1; i < num_analysis_options_stats; i++)
        EXPECT_FALSE(representativeregion.analysis_options_stats_yn[i]);
    int num_analysis_options_per_grain_stats = representativeregion.analysis_options_per_grain_stats_yn.size();
    for (int i = 0; i < num_analysis_options_per_grain_stats; i++) {
        if ((i == 1) || (i == 6))
            EXPECT_TRUE(representativeregion.analysis_options_per_grain_stats_yn[i]);
        else
            EXPECT_FALSE(representativeregion.analysis_options_per_grain_stats_yn[i]);
    }
    int num_analysis_options_per_z_stats = representativeregion.analysis_options_per_z_stats_yn.size();
    for (int i = 0; i < num_analysis_options_per_z_stats; i++)
        EXPECT_FALSE(representativeregion.analysis_options_per_z_stats_yn[i]);
    EXPECT_TRUE(representativeregion.print_per_grain_stats_yn);
    EXPECT_TRUE(representativeregion.print_pole_figure_yn);
    EXPECT_FALSE(representativeregion.print_exaconstit_yn);
    EXPECT_TRUE(representativeregion.print_inverse_pole_figure_map_yn);
}

// For a 3D domain, obtain a vector of the grain IDs that appear in a representative portion, and the associated grain
// statistics from the representative region
void testCollectGrainStats() {

    // Write out example data
    writeTestVolume();

    const int nx = 5; // Representative region spans 0-4 (entire domain in X)
    const int ny = 11; // Representative region spans 1-10 (one less than domain size in Y)
    const int nz = 12; // Representative region spans 0-9 (two less than domain size in Z)
    const double deltax = 1.25 * Kokkos::pow(10, -6);
    std::vector<double> xyz_bounds = {0.0, 0.0, 0.0, nx * deltax, ny * deltax, nz * deltax};

    // View for storing grain ID data
    Kokkos::View<int ***, Kokkos::HostSpace> grain_id(Kokkos::ViewAllocateWithoutInitializing("grain_id"), nz, nx, ny);
    Kokkos::View<short ***, Kokkos::HostSpace> phase_id(Kokkos::ViewAllocateWithoutInitializing("phase_id"), nz, nx,
                                                        ny);

    // Assign grain ID using the Z coordinate
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                grain_id(k, i, j) = k;
            }
        }
    }

    // Representative region creation
    std::ifstream analysis_data_stream("TestVolume.json");
    nlohmann::json analysis_data = nlohmann::json::parse(analysis_data_stream);
    RepresentativeRegion representativeregion(analysis_data, "RepresentativeVolume", nx, ny, nz, deltax, xyz_bounds,
                                              grain_id, phase_id, false);

    // Check results of grain ID vector (50 of each grain ID should exist)
    EXPECT_EQ(representativeregion.region_size_cells, nx * (ny - 1) * (nz - 2));
    for (int n = 0; n < representativeregion.region_size_cells; n++) {
        EXPECT_EQ(representativeregion.grain_id_vector[n], n / 50);
    }

    // Check number of grains and unique grain IDs
    EXPECT_EQ(representativeregion.number_of_grains, 10);
    for (int n = 0; n < representativeregion.number_of_grains; n++) {
        EXPECT_EQ(representativeregion.unique_grain_id_vector[n], n);
    }

    // Check against expected grain sizes, in cubic microns (each grain spans 50 cells)
    for (int n = 0; n < representativeregion.number_of_grains; n++) {
        float grain_size_microns_expected = static_cast<float>(50 * Kokkos::pow(deltax, 3) * Kokkos::pow(10, 18));
        EXPECT_FLOAT_EQ(representativeregion.grain_size_vector_microns[n], grain_size_microns_expected);
    }

    // Obtain the grain extents
    // Extent of each grain in X
    std::vector<float> grain_extent_x(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_x, grain_id, "X", deltax);
    // Extent of each grain in Y
    std::vector<float> grain_extent_y(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_y, grain_id, "Y", deltax);
    // Extent of each grain in Z
    std::vector<float> grain_extent_z(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_z, grain_id, "Z", deltax);
    // Check grain extents
    for (int n = 0; n < representativeregion.number_of_grains; n++) {
        // Expected value should be in microns
        EXPECT_FLOAT_EQ(grain_extent_x[n], nx * deltax * Kokkos::pow(10, 6));
        EXPECT_FLOAT_EQ(grain_extent_y[n], (ny - 1) * deltax * Kokkos::pow(10, 6));
        EXPECT_FLOAT_EQ(grain_extent_z[n], deltax * Kokkos::pow(10, 6));
    }
}

// For a 3D domain periodic in X and Y, ensure accurate calculations of grain extents in x, y, z
void testCollectGrainStats_Periodic() {

    const int nx = 10;
    const int ny = 20;
    const int nz = 10;
    // Write out example data
    writeTestVolume_FullDomain(nx, ny, nz);

    const double deltax = 1 * Kokkos::pow(10, -6);
    // Representative region spand entire volume
    std::vector<double> xyz_bounds = {0.0, 0.0, 0.0, nx * deltax, ny * deltax, nz * deltax};

    // View for storing grain ID data
    Kokkos::View<int ***, Kokkos::HostSpace> grain_id(Kokkos::ViewAllocateWithoutInitializing("grain_id"), nz, nx, ny);
    Kokkos::View<short ***, Kokkos::HostSpace> phase_id(Kokkos::ViewAllocateWithoutInitializing("phase_id"), nz, nx,
                                                        ny);

    // Grain 1: Doesn't cross X or Y domain boundaries
    int i_grain_nonperiodic[5] = {3, 4, 5, 6, 7};
    for (int k = 0; k < 2; k++) {
        for (int ii = 0; ii < 5; ii++) {
            int i = i_grain_nonperiodic[ii];
            for (int j = 10; j < 18; j++) {
                grain_id(k, i, j) = 1;
            }
        }
    }

    // Grain 2: Crosses X boundary but not Y
    int i_grain_periodic[5] = {8, 9, 0, 1, 2};
    for (int k = 0; k < 2; k++) {
        for (int ii = 0; ii < 5; ii++) {
            int i = i_grain_periodic[ii];
            for (int j = 10; j < 18; j++) {
                grain_id(k, i, j) = 2;
            }
        }
    }

    // Grain 3: Crosses X and Y boundaries
    int j_grain_periodic[3] = {18, 19, 0};
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < nx; i++) {
            for (int jj = 0; jj < 3; jj++) {
                int j = j_grain_periodic[jj];
                grain_id(k, i, j) = 3;
            }
        }
    }

    // Grain 4: Remainder of domain at Z = 0, 1
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 1; j < 10; j++) {
                grain_id(k, i, j) = 4;
            }
        }
    }

    // Grain 5: Remainder of domain at Z > 1
    for (int k = 2; k < nz; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                grain_id(k, i, j) = 5;
            }
        }
    }

    // Representative region creation with periodic boundary in X and Y
    std::ifstream analysis_data_stream("TestVolume.json");
    nlohmann::json analysis_data = nlohmann::json::parse(analysis_data_stream);
    RepresentativeRegion representativeregion(analysis_data, "RepresentativeVolume", nx, ny, nz, deltax, xyz_bounds,
                                              grain_id, phase_id, true);

    // Obtain the grain extents
    // Extent of each grain in X
    std::vector<float> grain_extent_x(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_x, grain_id, "X", deltax);
    // Extent of each grain in Y
    std::vector<float> grain_extent_y(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_y, grain_id, "Y", deltax);
    // Extent of each grain in Z
    std::vector<float> grain_extent_z(representativeregion.number_of_grains);
    representativeregion.calcGrainExtent(grain_extent_z, grain_id, "Z", deltax);

    // Check grain extents in microns
    int expected_extent_x_cells[5] = {5, 5, nx, nx, nx};
    int expected_extent_y_cells[5] = {8, 8, 3, 9, ny};
    int expected_extent_z_cells[5] = {2, 2, 2, 2, nz - 2};
    for (int n = 0; n < 5; n++) {
        std::cout << n + 1 << std::endl;
        EXPECT_FLOAT_EQ(grain_extent_x[n], expected_extent_x_cells[n] * deltax * Kokkos::pow(10, 6));
        EXPECT_FLOAT_EQ(grain_extent_y[n], expected_extent_y_cells[n] * deltax * Kokkos::pow(10, 6));
        EXPECT_FLOAT_EQ(grain_extent_z[n], expected_extent_z_cells[n] * deltax * Kokkos::pow(10, 6));
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, representative_region) {
    testConstructRepresentativeRegion_Volume();
    testConstructRepresentativeRegion_Area();
    testCollectGrainStats();
    testCollectGrainStats_Periodic();
}
} // end namespace Test
