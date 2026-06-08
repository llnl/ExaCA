// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "GAutils.hpp"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

// Parse log file using json format
void parseLogFile(std::string log_file, int &nx, int &ny, int &nz, double &deltax, int &number_of_layers,
                  std::vector<double> &xyz_bounds, std::vector<std::string> &grain_unit_vector_file,
                  std::vector<std::string> &phase_names, int &num_phases, bool orientation_files_in_input) {

    std::ifstream input_data_stream;
    input_data_stream.open(log_file);
    nlohmann::json logdata = nlohmann::json::parse(input_data_stream);

    // X, Y, Z bounds of domain (in microns)
    xyz_bounds[0] = logdata["Domain"]["XBounds"][0];
    xyz_bounds[1] = logdata["Domain"]["YBounds"][0];
    xyz_bounds[2] = logdata["Domain"]["ZBounds"][0];
    xyz_bounds[3] = logdata["Domain"]["XBounds"][1];
    xyz_bounds[4] = logdata["Domain"]["YBounds"][1];
    xyz_bounds[5] = logdata["Domain"]["ZBounds"][1];
    // Cell size (in microns)
    deltax = logdata["Domain"]["CellSize"];
    // Number of cells per domain direction
    nx = logdata["Domain"]["Nx"];
    ny = logdata["Domain"]["Ny"];
    nz = logdata["Domain"]["Nz"];
    std::string simulation_type = logdata["SimulationType"];
    // Directional, Spot, SingleGrain problem types are single layer
    if (simulation_type == "FromFinch" || simulation_type == "FromFile")
        number_of_layers = logdata["Domain"]["NumberOfLayers"];
    else
        number_of_layers = 1;

    if (orientation_files_in_input)
        std::cout << "Note: orientation filename specified in log file will be used, overriding value from analysis "
                     "input file"
                  << std::endl;

    std::string crystal_orientation_input_label;
    if (logdata.contains("GrainOrientationFile"))
        crystal_orientation_input_label = "GrainOrientationFile";
    else
        crystal_orientation_input_label = "CrystalOrientationFile";
    if (logdata["PhaseName"].size() == 2) {
        phase_names.push_back(logdata["PhaseName"][0]);
        phase_names.push_back(logdata["PhaseName"][1]);
        // List of grain unit vector files should be size 2 and consist of two separate files (for an orientation
        // transformation on solidification) or the same file twice. As of now, the second orientation file is not used
        // as materials without a transformation on solidification should only have one orientation file, and materials
        // with a transformation on solidification will have all final grain orientations map to a value from the first
        // file
        if (logdata[crystal_orientation_input_label].size() == 2) {
            grain_unit_vector_file.push_back(logdata[crystal_orientation_input_label][0]);
            grain_unit_vector_file.push_back(logdata[crystal_orientation_input_label][1]);
        }
        else {
            grain_unit_vector_file.push_back(logdata[crystal_orientation_input_label]);
            grain_unit_vector_file.push_back(logdata[crystal_orientation_input_label]);
        }
        num_phases = 2;
    }
    else {
        if (logdata.contains("PhaseName"))
            phase_names.push_back(logdata["PhaseName"]);
        else
            phase_names.push_back("default");
        num_phases = 1;
        grain_unit_vector_file.push_back(logdata[crystal_orientation_input_label]);
    }
    input_data_stream.close();
}

// Reads and ignored ASCII field data - each line contains data for a specific z coordinate
void readIgnoreASCIIField(std::ifstream &input_data_stream, int nz) {
    std::string line;
    for (int i = 0; i < nz; i++)
        getline(input_data_stream, line);
}

// Get the scaling factor to convert a number of cells into a length, area, or volume
double convertToMicrons(double deltax, std::string region_type) {
    double cell_to_size_scaling;
    if (region_type == "volume")
        cell_to_size_scaling = Kokkos::pow(deltax, 3) * Kokkos::pow(10, 18);
    else if (region_type == "area")
        cell_to_size_scaling = Kokkos::pow(deltax, 2) * Kokkos::pow(10, 12);
    else if (region_type == "length")
        cell_to_size_scaling = deltax * Kokkos::pow(10, 6);
    else
        throw std::runtime_error("Error: unknown region type");
    return cell_to_size_scaling;
}

// Get the scaling factor to convert a length, area, or volume into a number of cells
double convertToCells(double deltax, std::string region_type) {
    double size_to_cell_scaling;
    if (region_type == "volume")
        size_to_cell_scaling = 1.0 / (pow(deltax, 3) * Kokkos::pow(10, 18));
    else if (region_type == "area")
        size_to_cell_scaling = 1.0 / (pow(deltax, 2) * Kokkos::pow(10, 12));
    else if (region_type == "length")
        size_to_cell_scaling = 1.0 / (deltax * Kokkos::pow(10, 6));
    else
        throw std::runtime_error("Error: unknown region type");
    return size_to_cell_scaling;
}

// Print the string "output" to both output file streams
void dualPrint(std::string output, std::ostream &stream1, std::ostream &stream2) {
    stream1 << output;
    stream2 << output;
}
