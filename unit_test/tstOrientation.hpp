// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAconfig.hpp"
#include "CAorientation.hpp"
#include "CAparsefiles.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// orientation init tests
//---------------------------------------------------------------------------//
void testOrientationInit_Vectors() {

    using memory_space = TEST_MEMSPACE;

    std::string crystal_orientation_file_s = checkFileInstalled("CrystalOrientationVectors.csv", 0);

    // Initialize grain orientations - unit vector form only
    int id = 0;
    std::vector<std::string> crystal_orientation_file = {crystal_orientation_file_s};
    Orientation<memory_space> orientation(id, crystal_orientation_file, false);

    // Check results for first 2 orientations (first 18 values in the file)
    EXPECT_EQ(orientation.n_crystal_orientations, 10000);

    std::vector<float> expected_grain_unit_vector = {0.848294,  0.493303,  0.19248,  -0.522525, 0.720911,  0.455253,
                                                     0.0858167, -0.486765, 0.869308, 0.685431,  0.188182,  0.7034,
                                                     -0.468504, 0.85348,   0.228203, -0.557394, -0.485963, 0.673166};
    auto grain_unit_vector_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), orientation.grain_unit_vector);
    for (int n = 0; n < 18; n++) {
        EXPECT_FLOAT_EQ(grain_unit_vector_host(n, 0), expected_grain_unit_vector[n]);
    }
}

void testOrientationInit_Angles() {

    using memory_space = TEST_MEMSPACE;

    std::string crystal_orientation_file_s = checkFileInstalled("CrystalOrientationVectors.csv", 0);
    std::vector<std::string> crystal_orientation_file = {crystal_orientation_file_s};

    // Initialize grain orientations - unit vector form and data from CrystalOrientationEulerAnglesBungeZXZ.csv should
    // be read
    int id = 0;
    Orientation<memory_space> orientation(id, crystal_orientation_file, true);

    // Check results
    EXPECT_EQ(orientation.n_crystal_orientations, 10000);

    // Check first two orientations (first 6 values in the file)
    std::vector<float> expected_euler_angles = {9.99854, 29.62172, 22.91854, 311.08350, 47.68814, 72.02547};
    for (int n = 0; n < 6; n++) {
        EXPECT_FLOAT_EQ(orientation.grain_bunge_euler_host(n, 0), expected_euler_angles[n]);
    }
}

void testConvertGrainIDForBuffer() {
    using memory_space = TEST_MEMSPACE;
    using view_type = Kokkos::View<int *, memory_space>;

    // Create a list of integer grain ID values
    // Test positive and negative values
    int n_grain_id_values = 7;
    int total_n_grain_id_values = 2 * n_grain_id_values;
    std::vector<int> grain_id_v(total_n_grain_id_values);
    grain_id_v[0] = 1;
    grain_id_v[1] = 2;
    grain_id_v[2] = 3;
    grain_id_v[3] = 10000;
    grain_id_v[4] = 10001;
    grain_id_v[5] = 19999;
    grain_id_v[6] = 30132129;
    for (int n = n_grain_id_values; n < total_n_grain_id_values; n++) {
        grain_id_v[n] = -grain_id_v[n - n_grain_id_values];
    }
    view_type grain_id(Kokkos::ViewAllocateWithoutInitializing("GrainID"), total_n_grain_id_values);
    auto grain_id_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), grain_id);
    for (int n = 0; n < total_n_grain_id_values; n++) {
        grain_id_host(n) = grain_id_v[n];
    }
    grain_id = Kokkos::create_mirror_view_and_copy(memory_space(), grain_id_host);

    int n_crystal_orientations = 10000;

    // Check that these were converted to their components and back correctly
    view_type grain_id_converted(Kokkos::ViewAllocateWithoutInitializing("grain_id_converted"),
                                 total_n_grain_id_values);

    Kokkos::parallel_for(
        "TestInitGrainIDs", total_n_grain_id_values, KOKKOS_LAMBDA(const int &n) {
            int my_grain_orientation = getCrystalOrientation(grain_id(n), n_crystal_orientations, false);
            int my_grain_number = getGrainNumber(grain_id(n), n_crystal_orientations);
            grain_id_converted[n] = getGrainID(my_grain_orientation, my_grain_number, n_crystal_orientations);
        });
    auto grain_id_converted_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), grain_id_converted);
    for (int n = 0; n < total_n_grain_id_values; n++)
        EXPECT_EQ(grain_id_v[n], grain_id_converted_host(n));
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, orientation) {
    testOrientationInit_Vectors();
    testOrientationInit_Angles();
    testConvertGrainIDForBuffer();
}
} // end namespace Test
