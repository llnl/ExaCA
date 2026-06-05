// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_ORIENTATION_HPP
#define EXACA_ORIENTATION_HPP

#include "CAparsefiles.hpp"

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

// Data representing the crystallographic orientations corresponding to GrainID values
template <typename MemorySpace>
struct Orientation {

    using memory_space = MemorySpace;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_float_host = typename view_type_float::host_mirror_type;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_int_host = typename view_type_int::host_mirror_type;
    using view_type_float_2d = Kokkos::View<float **, memory_space>;
    using view_type_float_2d_host = typename view_type_float_2d::host_mirror_type;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Grain orientation data views - either rotation matrices (9 vals per orientation), Euler angles (3 vals per
    // orientation), or RGB values (3 vals per orientation). Unit vectors are stored on the device and no host copy is
    // maintained in the struct, Euler and RGB representations are only used on the host and no device copy is
    // maintained
    int n_crystal_orientations;
    // Second dimension unused if second phase shares grain orientation with first, or the material only contains 1
    // phase
    view_type_float_2d grain_unit_vector;
    view_type_float_2d_host grain_bunge_euler_host, grain_rgb_ipfz_host;
    int _num_phases;
    bool _solidification_transformation_yn;

    Orientation(const int id, const std::vector<std::string> grain_unit_vector_file, const bool init_euler_rgb_vals,
                unsigned long rng_seed = 0.0, const int num_phases = 1,
                const bool solidification_transformation_yn = false)
        : grain_unit_vector(
              view_type_float_2d(Kokkos::ViewAllocateWithoutInitializing("grain_unit_vector"), 1, num_phases))
        , grain_bunge_euler_host(
              view_type_float_2d_host(Kokkos::ViewAllocateWithoutInitializing("grain_bunge_euler_host"), 1, num_phases))
        , grain_rgb_ipfz_host(
              view_type_float_2d_host(Kokkos::ViewAllocateWithoutInitializing("grain_rgb_ipfz_host"), 1, num_phases))
        , _num_phases(num_phases)
        , _solidification_transformation_yn(solidification_transformation_yn) {

        // Get unit vectors from the grain orientations file (9 vals per line) and store in temporary host view
        view_type_float_2d_host grain_unit_vector_host_ =
            getOrientationsFromFile(grain_unit_vector_file, 9, false, rng_seed);
        // Copy unit vector data from temporary host view
        grain_unit_vector = Kokkos::create_mirror_view_and_copy(memory_space(), grain_unit_vector_host_);

        // If needed, repeat this process for Euler and RGB representation of the orientations
        if (init_euler_rgb_vals) {
            // Other orientation files should start with the strings GrainOrientationEulerAnglesBungeZXZ and
            // GrainOrientationRGB_IPF-Z, but be proceeded with a unique string (such as _1e6.csv) that matches the
            // corresponding file of grain unit vectors. In the case of using the default file
            // GrainOrientationVectors.csv, this will be an empty string
            getEulerIPFZFilenames(grain_unit_vector_file, rng_seed);
        }
        if (id == 0)
            std::cout << "Done with orientation initialization " << std::endl;
    }

    // Get grain orientations from the specified file and the number of grain orientations and return a host view
    // containing the data
    view_type_float_2d_host getOrientationsFromFile(const std::vector<std::string> orientation_file,
                                                    const int vals_per_line, const bool check_n_orientations,
                                                    unsigned long rng_seed) {

        // Resize view for storing grain orientations read from file based on a guess for n_crystal_orientations (10000
        // used here)
        view_type_float_2d_host orientation_data_host(Kokkos::ViewAllocateWithoutInitializing("orientation_data_host"),
                                                      vals_per_line * 10000, _num_phases);
        for (int phase_num = 0; phase_num < _num_phases; phase_num++) {
            // Read file of grain orientations
            std::ifstream orientation_input_stream;
            orientation_input_stream.open(orientation_file[phase_num]);

            // Line 1 is the number of orientation values to read (if check_n_orientations is false, store this value as
            // n_orientations in the struct. Otherwise, read this value and check that it matches n_orientations)
            std::string n_crystal_orientations_read;
            getline(orientation_input_stream, n_crystal_orientations_read);
            // If reading 2 files, number of grain orientations must match
            if ((check_n_orientations) || (phase_num == 1)) {
                int n_crystal_orientations_check = getInputInt(n_crystal_orientations_read);
                if (n_crystal_orientations != n_crystal_orientations_check)
                    throw std::runtime_error("Error: The number of orientations given on the first line of all "
                                             "orientation files must match");
            }
            else
                n_crystal_orientations = getInputInt(n_crystal_orientations_read);

            // Resize view for storing grain orientations read from file based on the known value for
            // n_crystal_orientations
            Kokkos::resize(orientation_data_host, vals_per_line * n_crystal_orientations, _num_phases);

            // Populate data structure for grain orientation data and return the view
            for (int i = 0; i < n_crystal_orientations; i++) {
                std::vector<std::string> parsed_line(vals_per_line);
                std::string read_line;
                if (!getline(orientation_input_stream, read_line))
                    break;
                splitString(read_line, parsed_line, vals_per_line);
                // Place the 3 grain orientation angles or 9 rotation matrix components into the orientation data view
                for (int comp = 0; comp < vals_per_line; comp++) {
                    orientation_data_host(vals_per_line * i + comp, phase_num) = getInputFloat(parsed_line[comp]);
                }
            }
            orientation_input_stream.close();
        }

        // If there is an orientation transformation on solidification and only a single grain orientation file was
        // given, create a shuffled version of the primary phase orientations (using the random seed) to use as the
        // secondary phase orientations
        if (_num_phases == 2) {
            if (orientation_file[0] == orientation_file[1]) {
                std::vector<int> orientation_permutation_vector(n_crystal_orientations);
                std::mt19937_64 orientation_generator(rng_seed);
                for (int i = 0; i < n_crystal_orientations; i++)
                    orientation_permutation_vector[i] = i;
                std::shuffle(orientation_permutation_vector.begin(), orientation_permutation_vector.end(),
                             orientation_generator);
                for (int i = 0; i < n_crystal_orientations; i++) {
                    for (int comp = 0; comp < vals_per_line; comp++)
                        orientation_data_host(vals_per_line * i + comp, 1) =
                            orientation_data_host(vals_per_line * orientation_permutation_vector[i] + comp, 1);
                }
            }
        }
        return orientation_data_host;
    }

    // Get names of and read other orientation files
    void getEulerIPFZFilenames(const std::vector<std::string> grain_unit_vector_file, unsigned long rng_seed) {

        std::vector<std::string> euler_angles_filename, rgb_filename;
        for (int phase_num = 0; phase_num < _num_phases; phase_num++) {
            std::string grain_unit_vector_file_p = grain_unit_vector_file[phase_num];
            std::string euler_angles_filename_p, rgb_filename_p;
            if (grain_unit_vector_file_p.find("CrystalOrientationVectors.csv") != std::string::npos) {
                // Default files for euler angles and RGB mapping
                euler_angles_filename_p = "CrystalOrientationEulerAnglesBungeZXZ.csv";
                rgb_filename_p = "CrystalOrientationRGB_IPF-Z.csv";
            }
            else {
                // Custom files for euler angles and RGB mapping based on rotation filename
                std::size_t baseorientation_startpos = grain_unit_vector_file_p.find_last_of("/");
                std::size_t endpos = grain_unit_vector_file_p.find_last_of(".");
                std::string pathtoorientations = grain_unit_vector_file_p.substr(0, baseorientation_startpos + 1);
                std::string baseorientationname = grain_unit_vector_file_p.substr(
                    baseorientation_startpos + 1, endpos - baseorientation_startpos - 1);
                std::size_t customname_startpos = baseorientationname.find_last_of("_");
                std::string customname =
                    baseorientationname.substr(customname_startpos + 1, endpos - customname_startpos - 1);
                euler_angles_filename_p =
                    pathtoorientations + "CrystalOrientationEulerAnglesBungeZXZ_" + customname + ".csv";
                rgb_filename_p = pathtoorientations + "CrystalOrientationRGB_IPF-Z_" + customname + ".csv";
            }
            // Full path to install location was already given for the rotations file, need to check install location
            // for other files and ensure they are not empty
            euler_angles_filename_p = checkFileInstalled(euler_angles_filename_p, 0);
            rgb_filename_p = checkFileInstalled(rgb_filename_p, 0);
            checkFileNotEmpty(euler_angles_filename_p);
            checkFileNotEmpty(rgb_filename_p);
            euler_angles_filename.push_back(euler_angles_filename_p);
            rgb_filename.push_back(rgb_filename_p);
        }
        std::cout << "Reading " << euler_angles_filename[0] << " and " << rgb_filename[0] << std::endl;

        // Read and store additional orientation data in the appropriate views
        grain_bunge_euler_host = getOrientationsFromFile(euler_angles_filename, 3, true, rng_seed);
        grain_rgb_ipfz_host = getOrientationsFromFile(rgb_filename, 3, true, rng_seed);
    }

    // Create a view of size "n_crystal_orientations" of the misorientation of each possible grain orientation with the
    // X, Y, or Z directions (dir = 0, 1, or 2, respectively)
    view_type_float_2d_host misorientationCalc(const int dir) {

        view_type_float_2d_host crystal_misorientation(Kokkos::ViewAllocateWithoutInitializing("CrystalMisorientation"),
                                                       n_crystal_orientations, _num_phases);
        view_type_float_2d_host grain_unit_vector_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grain_unit_vector);
        for (int phase_num = 0; phase_num < _num_phases; phase_num++) {
            // Find the smallest possible misorientation between the specified direction, and this grain orientations' 6
            // possible 001 directions (where 54.7356 degrees is the largest possible misorientation between a 001 and a
            // given cardinal direction for the cubic crystal system)
            for (int n = 0; n < n_crystal_orientations; n++) {
                float misorientation_angle_min = 54.7356;
                for (int ll = 0; ll < 3; ll++) {
                    float misorientation =
                        Kokkos::abs((180 / M_PI) *
                                    Kokkos::acos(Kokkos::abs(grain_unit_vector_host(9 * n + 3 * ll + dir, phase_num))));
                    if (misorientation < misorientation_angle_min) {
                        misorientation_angle_min = misorientation;
                    }
                }
                crystal_misorientation(n, phase_num) = misorientation_angle_min;
            }
        }
        return crystal_misorientation;
    }
};

// Inline functions
// Get the grain ID from the repeat number of a grain from a given grain ID and the number of possible orientations
KOKKOS_INLINE_FUNCTION int getGrainID(const int my_crystal_orientation, const int my_grain_number,
                                      const int n_crystal_orientations) {
    int my_grain_id = n_crystal_orientations * (Kokkos::abs(my_grain_number) - 1) + my_crystal_orientation;
    if (my_grain_number < 0)
        my_grain_id = -my_grain_id;
    return my_grain_id;
}

// Get the orientation of a grain's <100> from a given grain ID and the number of possible orientations
// By default, start indexing at 0 (GrainID of 1 has Orientation number 0), optionally starting at 1
// GrainID of 0 is a special case - has orientation 0 no matter what (only used when reconstructing grain ID in
// getGrainID)
KOKKOS_INLINE_FUNCTION int getCrystalOrientation(const int my_grain_id, const int n_crystal_orientations,
                                                 bool start_at_zero = true) {
    int my_orientation;
    if (my_grain_id == 0)
        my_orientation = 0;
    else {
        my_orientation = (Kokkos::abs(my_grain_id) - 1) % n_crystal_orientations;
        if (!(start_at_zero))
            my_orientation++;
    }
    return my_orientation;
}

// Get the repeat number for the orientation of a grain with a given grain ID
// 1, 2, 3... or -1, -2, -3...
KOKKOS_INLINE_FUNCTION int getGrainNumber(int my_grain_id, int n_crystal_orientations) {
    int my_grain_number = (Kokkos::abs(my_grain_id) - 1) / n_crystal_orientations + 1;
    if (my_grain_id < 0)
        my_grain_number = -my_grain_number;
    return my_grain_number;
}

#endif
