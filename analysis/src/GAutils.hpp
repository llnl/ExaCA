// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef GA_UTIL_HPP
#define GA_UTIL_HPP

#include "CAparsefiles.hpp"

#include <Kokkos_Core.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// These are used in reading/parsing ExaCA microstructure data
void parseLogFile(std::string logfile, int &nx, int &ny, int &nz, double &deltax, int &number_of_layers,
                  std::vector<double> &xyz_bounds, std::vector<std::string> &grain_unit_vector_file,
                  std::vector<std::string> &phase_names, int &num_phases, bool orientation_files_in_input,
                  bool &is_periodic);
double convertToMicrons(double deltax, std::string region_type);
double convertToCells(double deltax, std::string region_type);
void dualPrint(std::string temp, std::ostream &stream1, std::ostream &stream2);
template <typename ReturnType, typename FirstType, typename SecondType>
ReturnType divideCast(FirstType int1, SecondType int2) {
    return static_cast<ReturnType>(int1) / static_cast<ReturnType>(int2);
}

template <typename ViewTypeInt3dHost, typename ViewTypeShort3dHost>
void initializeData(std::string microstructure_file, int nx, int ny, int nz, ViewTypeInt3dHost &grain_id,
                    ViewTypeShort3dHost &layer_id, ViewTypeShort3dHost &phase_id, bool &found_layer_id) {

    std::ifstream input_data_stream;
    input_data_stream.open(microstructure_file);
    if (!(input_data_stream))
        throw std::runtime_error("Error: Cannot find ExaCA microstructure file");
    std::string line;
    bool binary_vtk = false;
    // 8 header lines at beginning of Paraview file from header
    // 3rd line tells whether data is in ASCII or binary format
    for (int header_lines = 0; header_lines < 8; header_lines++) {
        getline(input_data_stream, line);
        if (header_lines == 2)
            if (line.find("BINARY") != std::string::npos)
                binary_vtk = true;
    }
    bool reading_vtk = true;
    bool found_grain_id = false;
    bool found_phase_id = false;
    while (reading_vtk) {
        // This line says which variable appears next in the file, along with its type
        // A blank line ends the file read
        std::string read_datatype_string, read_fieldname;
        getline(input_data_stream, line);
        if (line.empty())
            reading_vtk = false;
        else {
            // Thus far, every ExaCA field printed to VTK files has been one of these three types - ensure that this
            // field is one of them
            std::vector<std::string> possible_vtk_datatypes = {"short", "int", "float"};
            int num_possible_vtk_types = possible_vtk_datatypes.size();
            for (auto n = 0; n < num_possible_vtk_types; n++) {
                if (line.find(possible_vtk_datatypes[n]) != std::string::npos)
                    read_datatype_string = possible_vtk_datatypes[n];
            }
            if (read_datatype_string.empty())
                throw std::runtime_error("Error: Unknown data type for a field in the vtk file");

            // grain_id is only fields currently used by the analysis, other fields will be skipped/ignored
            std::vector<std::string> possible_fieldnames = {"GrainID", "LayerID", "PhaseID"};
            int num_possible_fieldnames = possible_fieldnames.size();
            for (auto n = 0; n < num_possible_fieldnames; n++) {
                if (line.find(possible_fieldnames[n]) != std::string::npos)
                    read_fieldname = possible_fieldnames[n];
            }
            if (read_fieldname.empty()) {
                std::cout << "Reading and ignoring field described by " << line << std::endl;
                // 1 more unused line
                getline(input_data_stream, line);
                if (binary_vtk) {
                    if (read_datatype_string == "short")
                        readIgnoreBinaryField<short>(input_data_stream, nx, ny, nz);
                    else if (read_datatype_string == "int")
                        readIgnoreBinaryField<int>(input_data_stream, nx, ny, nz);
                    else if (read_datatype_string == "float")
                        readIgnoreBinaryField<float>(input_data_stream, nx, ny, nz);
                }
                else
                    readIgnoreASCIIField(input_data_stream, nz);
            }
            else {
                // Valid fieldname to be read
                // 1 more unused line
                getline(input_data_stream, line);
                // Place grain_id data and (optionally) layer_id data
                if (read_fieldname == possible_fieldnames[0]) {
                    // grain_id data should be type int
                    if (read_datatype_string != "int")
                        throw std::runtime_error("Error: Field GrainID should be data of type int");
                    if (binary_vtk)
                        grain_id = readBinaryField<Kokkos::View<int ***, Kokkos::HostSpace>, int>(input_data_stream, nx,
                                                                                                  ny, nz, "GrainID");
                    else
                        grain_id = readASCIIField<Kokkos::View<int ***, Kokkos::HostSpace>>(input_data_stream, nx, ny,
                                                                                            nz, "GrainID");
                    found_grain_id = true;
                }
                else if (read_fieldname == possible_fieldnames[1]) {
                    // layer_id may be int or short, but is stored as type short
                    if ((read_datatype_string != "int") && (read_datatype_string != "short"))
                        throw std::runtime_error("Error: Field LayerID should be data of type int or short");
                    if (!binary_vtk)
                        layer_id = readASCIIField<Kokkos::View<short ***, Kokkos::HostSpace>>(input_data_stream, nx, ny,
                                                                                              nz, "LayerID");
                    else {
                        if (read_datatype_string == "int")
                            layer_id = readBinaryField<Kokkos::View<short ***, Kokkos::HostSpace>, int>(
                                input_data_stream, nx, ny, nz, "LayerID");
                        else
                            layer_id = readBinaryField<Kokkos::View<short ***, Kokkos::HostSpace>, short>(
                                input_data_stream, nx, ny, nz, "LayerID");
                    }
                    found_layer_id = true;
                }
                else if (read_fieldname == possible_fieldnames[2]) {
                    if (!binary_vtk)
                        phase_id = readASCIIField<Kokkos::View<short ***, Kokkos::HostSpace>>(input_data_stream, nx, ny,
                                                                                              nz, "PhaseID");
                    else {
                        phase_id = readBinaryField<Kokkos::View<short ***, Kokkos::HostSpace>, short>(
                            input_data_stream, nx, ny, nz, "PhaseID");
                    }
                    found_phase_id = true;
                }
                std::cout << "Data field " << read_fieldname << " read" << std::endl;
            }
        }
    }
    input_data_stream.close();
    if (!found_grain_id)
        throw std::runtime_error("Error: analysis requires the GrainID field to be present in the vtk file");
    if (!found_layer_id) {
        std::cout << "Note: LayerID not present in vtk data, analysis will not differentiate between cells that did "
                     "and did not undergo melting"
                  << std::endl;
        Kokkos::deep_copy(layer_id, 0);
    }
    // PhaseID defaults to zeros if not present in the file (assumed single as-solidified phase)
    if (!found_phase_id)
        Kokkos::deep_copy(phase_id, 0);
}

#endif
