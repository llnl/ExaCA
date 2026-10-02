// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_GRID_HPP
#define EXACA_GRID_HPP

#include "CAinputdata.hpp"
#include "CAparsefiles.hpp"

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Data for local (individual MPI ranks) and global (across all ranks) grids
struct Grid {

    // Global domain bounds, all layers
    int nx, ny, nz, domain_size_all_layers;
    double deltax, x_min, x_max, y_min, y_max, z_min, z_max;

    // Variables characterizing local processor grids relative to global domain
    // 1D decomposition in Y: Each MPI rank has a subset consisting of of ny_local cells, out of ny cells in Y
    // Each MPI rank's subdomain is offset by y_offset cells from the lower bound of the domain in Y
    int ny_local, y_offset;
    std::vector<int> ny_local_allranks, y_offset_allranks;
    // Periodicity in X and Y (only for directional solidification simulations)
    bool is_periodic = false;
    // Variables characterizing process IDs of neighboring MPI ranks on the grid
    // Positive Y/NegativeY directions are North/South
    int neighbor_rank_north, neighbor_rank_south;
    // Variables denoting whether or not each MPI rank's grid is at a global domain boundary
    bool at_north_boundary, at_south_boundary;

    // Multilayer problem information
    using view_type_double_host = Kokkos::View<double *, Kokkos::HostSpace>;
    view_type_double_host z_min_layer, z_max_layer;
    int number_of_layers, layer_height;
    double HT_deltax;

    // Current layer information for multilayer problems
    int z_layer_bottom, z_layer_top, nz_layer, domain_size, bottom_of_current_layer, top_of_current_layer;
    std::pair<int, int> layer_range;

    // Domain inputs from file
    DomainInputs _inputs;
    // Substrate inputs from file
    SubstrateInputs _s_inputs;
    // Temperature inputs from file
    TemperatureInputs _t_inputs;

    // Creates grid struct with uninitialized values, used in unit tests
    Grid(int number_of_layers_temp = 1)
        : z_min_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_min_layer"), number_of_layers_temp))
        , z_max_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_max_layer"), number_of_layers_temp)) {
        number_of_layers = number_of_layers_temp;
        ny_local_allranks.resize(1);
        y_offset_allranks.resize(1);
    };

    // Creates grid struct from Finch grid - currently only supports simulations where one set of temperature data is
    // used (either for single layer simulations or multilayer simulations repeating the same data with an layer height
    // offset in the Z direction)
    Grid(const int id, const int np, DomainInputs inputs, SubstrateInputs s_inputs, TemperatureInputs t_inputs,
         const double finch_cell_size, std::array<double, 3> global_low_corner,
         std::array<double, 3> global_high_corner, const double cell_size_tolerance = 1 * Kokkos::pow(10, -8))
        : z_min_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_min_layer"), inputs.number_of_layers))
        , z_max_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_max_layer"), inputs.number_of_layers))
        , number_of_layers(inputs.number_of_layers)
        , layer_height(inputs.layer_height)
        , _inputs(inputs)
        , _s_inputs(s_inputs)
        , _t_inputs(t_inputs) {

        // Ensure Finch and ExaCA cell sizes match
        deltax = _inputs.deltax;
        if (Kokkos::abs(finch_cell_size - deltax) > cell_size_tolerance)
            throw std::runtime_error("Error: ExaCA cell size must match cell size from Finch simulation");
        // Set ExaCA domain bounds using Finch domain bounds - extended in X or Y if considering additional translation
        // of Finch data
        x_min = global_low_corner[0];
        y_min = global_low_corner[1];
        z_min = global_low_corner[2];
        x_max = global_high_corner[0] + (_t_inputs.number_of_copies - 1) * _t_inputs.x_offset;
        y_max = global_high_corner[1] + (_t_inputs.number_of_copies - 1) * _t_inputs.y_offset;
        z_max = global_high_corner[2] + inputs.layer_height * (inputs.number_of_layers - 1) * deltax;
        // If domain bounds were specified, override domain bounds from the data in the files
        checkOverrideXYZBounds(id, _s_inputs, _t_inputs);
        nx = Kokkos::round((x_max - x_min) / deltax) + 1;
        ny = Kokkos::round((y_max - y_min) / deltax) + 1;
        nz = Kokkos::round((z_max - z_min) / deltax) + 1;
        for (int n = 0; n < inputs.number_of_layers; n++) {
            z_min_layer(n) = z_min + n * inputs.layer_height * deltax;
            z_max_layer(n) = global_high_corner[2] + n * inputs.layer_height * deltax;
        }
        // Domain decomposition
        decomposeDomain(id, np, "FromFinch");
        MPI_Barrier(MPI_COMM_WORLD);
        // Print domain info to console
        if (id == 0)
            printDomainBounds(np);
    }

    // If domain bounds were specified, override domain bounds from the data in the files
    void checkOverrideXYZBounds(const int id, SubstrateInputs _s_inputs, TemperatureInputs _t_inputs) {
        if (_t_inputs.use_fixed_x_bounds) {
            if (id == 0)
                std::cout << "Overriding the X bounds given by the temperature data (" << x_min << ", " << x_max
                          << ") with those given in the input file (" << _t_inputs.temperature_x_bounds[0] << ", "
                          << _t_inputs.temperature_x_bounds[1] << ")" << std::endl;
            x_min = _t_inputs.temperature_x_bounds[0];
            x_max = _t_inputs.temperature_x_bounds[1];
        }
        if (_t_inputs.use_fixed_y_bounds) {
            if (id == 0)
                std::cout << "Overriding the Y bounds given by the temperature data (" << y_min << ", " << y_max
                          << ") with those given in the input file (" << _t_inputs.temperature_y_bounds[0] << ", "
                          << _t_inputs.temperature_y_bounds[1] << ")" << std::endl;
            y_min = _t_inputs.temperature_y_bounds[0];
            y_max = _t_inputs.temperature_y_bounds[1];
        }
        if (_s_inputs.use_fixed_z_bounds) {
            // Check to ensure that the specified lower Z bound is smaller than the smallest Z with temperature data
            // (otherwise throw an error)
            if (z_min < _s_inputs.baseplate_bottom_z)
                throw std::runtime_error("Error: temperature data extends below the input value for BaseplateBottomZ");
            if (id == 0)
                std::cout << "Overriding the lower Z bound given by the temperature data (" << z_min
                          << ") with that given in the input file (" << _s_inputs.baseplate_bottom_z << ")"
                          << std::endl;
            z_min = _s_inputs.baseplate_bottom_z;
        }
    }

    // Constructor for grid used in ExaCA for problem types other than FromFinch
    Grid(const std::string simulation_type, const int id, const int np, const int number_of_layers_temp,
         DomainInputs inputs, SubstrateInputs s_inputs, TemperatureInputs t_inputs)
        : z_min_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_min_layer"), number_of_layers_temp))
        , z_max_layer(
              view_type_double_host(Kokkos::ViewAllocateWithoutInitializing("z_max_layer"), number_of_layers_temp))
        , _inputs(inputs)
        , _s_inputs(s_inputs)
        , _t_inputs(t_inputs) {

        // Copy from inputs structs
        deltax = _inputs.deltax;
        layer_height = _inputs.layer_height;
        layer_height = _inputs.layer_height;
        number_of_layers = number_of_layers_temp;
        is_periodic = _inputs.is_periodic;

        // Obtain global domain bounds
        // For problem type FromFile, need to parse all temperature data files to obtain domain bounds for
        // each layer and for the multilayer domain
        if (simulation_type == "FromFile")
            findXYZBounds(id, s_inputs, t_inputs);
        else {
            // Copy inputs from inputs struct into grid struct
            nx = _inputs.nx;
            ny = _inputs.ny;
            nz = _inputs.nz;
            if (simulation_type == "Spot") {
                // Domain edge in X and Y is at 0
                x_min = 0.0;
                y_min = 0.0;
                x_max = (nx - 1) * deltax;
                y_max = (ny - 1) * deltax;
                // Domain top is at 0
                z_min = -(nz - 1) * deltax;
                z_min_layer(0) = z_min;
                z_max = 0.0;
                z_max_layer(0) = z_max;
            }
            else {
                // Domain origin is placed at 0
                x_min = 0.0;
                y_min = 0.0;
                z_min = 0.0;
                z_min_layer(0) = z_min;
                x_max = (nx - 1) * deltax;
                y_max = (ny - 1) * deltax;
                z_max = (nz - 1) * deltax;
                z_max_layer(0) = z_max;
                if (is_periodic) {
                    x_min = x_min - deltax;
                    x_max = x_max + deltax;
                    nx += 2;
                    y_min = y_min - deltax;
                    y_max = y_max + deltax;
                }
            }
        }
        // Domain decomposition
        decomposeDomain(id, np, simulation_type);
        // After decomposition, add padding to global domain at +/-Y edges for periodic boundary if needed
        if (is_periodic)
            ny += 2;
        // Print domain info to console
        if (id == 0)
            printDomainBounds(np);
    };

    // Called on rank 0 only
    void printDomainBounds(const int np) {
        std::cout << "Domain size: " << nx << " by " << ny << " by " << nz << std::endl;
        std::cout << "X Limits of domain: " << x_min << " and " << x_max << std::endl;
        std::cout << "Y Limits of domain: " << y_min << " and " << y_max << std::endl;
        std::cout << "Z Limits of domain: " << z_min << " and " << z_max << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "Mesh initialized: initial domain size is " << nz_layer << " out of " << nz
                  << " total cells in the Z direction" << std::endl;
        for (int pid = 0; pid < np; pid++)
            std::cout << "Rank " << pid << " spans Y = " << y_offset_allranks[pid] << " through "
                      << y_offset_allranks[pid] + ny_local_allranks[pid] - 1 << std::endl;
        std::cout << "================================================================" << std::endl;
    }

    // Perform domain decomposition and initialize first layer's grid
    void decomposeDomain(const int id, const int np, std::string simulation_type) {

        // Decompose the domain into subdomains on each MPI rank: Calculate ny_local and y_offset for each rank, where
        // each subdomain contains "ny_local" in Y, offset from the full domain origin by "y_offset" cells in Y
        // (previously "DomainDecomposition" in CAinitialize.cpp) First, compare total MPI ranks to total Y cells.
        if (np > static_cast<double>(ny) / 2.0)
            throw std::runtime_error("Error: Must have at least 2 cells in Y (decomposition direction) per MPI rank.");

        // The following was previously performed in "DomainDecomposition" in CAinitialize.cpp:
        // Get neighboring ranks to each direction and set boundary indicator variables
        neighbor_rank_north = getNeighborRankNorth(id, np);
        neighbor_rank_south = getNeighborRankSouth(id, np);
        at_north_boundary = getAtNorthBoundary();
        at_south_boundary = getAtSouthBoundary();

        // Determine, for each MPI process id, the local grid size in y (and the offset in y relative to the overall
        // simulation domain)
        y_offset = getYOffset(id, np);
        ny_local = getNyLocal(id, np);

        // Add halo regions with a width of 1 in +/- Y if this MPI rank is not as a domain boundary in said direction
        addHalo(np);
        // Domain size across all ranks and all layers
        domain_size_all_layers = getDomainSizeAllLayers();
        // Rank 0 stores each MPI rank's subdomain bounds in Y
        ny_local_allranks.resize(np);
        y_offset_allranks.resize(np);
        MPI_Gather(&ny_local, 1, MPI_INT, ny_local_allranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&y_offset, 1, MPI_INT, y_offset_allranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Bounds of layer 0: Z coordinates span z_layer_bottom-z_layer_top, inclusive (functions previously in
        // CAinitialize.cpp and CAcelldata.hpp)
        z_layer_bottom = calcZLayerBottom(simulation_type, 0);
        z_layer_top = calcZLayerTop(simulation_type, 0);
        nz_layer = calcNzLayer(id, 0);
        domain_size = calcDomainSize(); // Number of cells in the current layer on this MPI rank
        bottom_of_current_layer = getBottomOfCurrentLayer();
        top_of_current_layer = getTopOfCurrentLayer();
        layer_range = std::make_pair(bottom_of_current_layer, top_of_current_layer);
    }

    // Read x, y, z coordinates in tempfile_thislayer (temperature file in either an ASCII or binary format) and return
    // the min and max values. This is not used by the temperature struct (no temperature info is parsed or stored), and
    // is only used to find the bounds of the simulation domain
    std::array<double, 6> parseTemperatureCoordinateMinMax(std::string tempfile_thislayer, bool binary_input_data) {

        std::array<double, 6> xyz_min_max;
        std::ifstream temperature_filestream;
        temperature_filestream.open(tempfile_thislayer);
        std::size_t vals_per_line;

        // Binary temperature data should contain only the six columns of interest
        // Comma-separated double type values may contain additional columns after the 6 used by ExaCA
        if (binary_input_data)
            vals_per_line = 6;
        else {
            // Read the header line data
            // Make sure the first line contains all required column names: x, y, z, tm, tl, cr
            std::string header_line;
            getline(temperature_filestream, header_line);
            vals_per_line = checkForHeaderValues(header_line);
        }

        // Units are assumed to be in meters, meters, seconds, seconds, and K/second
        int xyz_point_count_estimate = 1000000;
        std::vector<double> x_coordinates(xyz_point_count_estimate), y_coordinates(xyz_point_count_estimate),
            z_coordinates(xyz_point_count_estimate);
        long unsigned int xyz_point_counter = 0;
        if (binary_input_data) {
            while (!temperature_filestream.eof()) {
                // Get x from the binary string, or, if no data is left, exit the file read
                double x_value = readBinaryData<double>(temperature_filestream);
                if (!(temperature_filestream))
                    break;
                // Store the x value that was read, and parse the y and z values
                x_coordinates[xyz_point_counter] = x_value;
                y_coordinates[xyz_point_counter] = readBinaryData<double>(temperature_filestream);
                z_coordinates[xyz_point_counter] = readBinaryData<double>(temperature_filestream);
                // Ignore the tm, tl, cr values associated with this x, y, z
                unsigned char temp[3 * sizeof(double)];
                temperature_filestream.read(reinterpret_cast<char *>(temp), 3 * sizeof(double));
                xyz_point_counter++;
                if (xyz_point_counter == x_coordinates.size()) {
                    x_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                    y_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                    z_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                }
            }
        }
        else {
            while (!temperature_filestream.eof()) {
                std::vector<std::string> parsed_line(3); // Get x, y, z - ignore tm, tl, cr
                std::string read_line;
                if (!getline(temperature_filestream, read_line))
                    break;
                splitString(read_line, parsed_line, vals_per_line);
                // Only get x, y, and z values from ParsedLine
                x_coordinates[xyz_point_counter] = getInputDouble(parsed_line[0]);
                y_coordinates[xyz_point_counter] = getInputDouble(parsed_line[1]);
                z_coordinates[xyz_point_counter] = getInputDouble(parsed_line[2]);
                xyz_point_counter++;
                if (xyz_point_counter == x_coordinates.size()) {
                    x_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                    y_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                    z_coordinates.resize(xyz_point_counter + xyz_point_count_estimate);
                }
            }
        }
        if (xyz_point_counter == 0) {
            std::string error_message = "Error: File " + tempfile_thislayer + " contained no data";
            throw std::runtime_error(error_message);
        }
        x_coordinates.resize(xyz_point_counter);
        y_coordinates.resize(xyz_point_counter);
        z_coordinates.resize(xyz_point_counter);
        temperature_filestream.close();

        // Min/max x, y, and z coordinates from this layer's data
        xyz_min_max[0] = *min_element(x_coordinates.begin(), x_coordinates.end());
        xyz_min_max[1] = *max_element(x_coordinates.begin(), x_coordinates.end());
        xyz_min_max[2] = *min_element(y_coordinates.begin(), y_coordinates.end());
        xyz_min_max[3] = *max_element(y_coordinates.begin(), y_coordinates.end());
        xyz_min_max[4] = *min_element(z_coordinates.begin(), z_coordinates.end());
        xyz_min_max[5] = *max_element(z_coordinates.begin(), z_coordinates.end());
        return xyz_min_max;
    }

    // For simulation type FromFile, obtain the physical XYZ bounds of the domain by reading temperature data files and
    // parsing the coordinates (does not store temperature data points). Previously in CAinitialize.cpp
    void findXYZBounds(const int id, SubstrateInputs _s_inputs, TemperatureInputs _t_inputs) {

        x_min = std::numeric_limits<double>::max();
        y_min = std::numeric_limits<double>::max();
        z_min = std::numeric_limits<double>::max();
        x_max = std::numeric_limits<double>::lowest();
        y_max = std::numeric_limits<double>::lowest();
        z_max = std::numeric_limits<double>::lowest();

        // Read the first temperature file
        std::ifstream first_temperature_file;
        first_temperature_file.open(_t_inputs.temp_paths[0]);
        std::string first_line_first_file;
        // Header line
        getline(first_temperature_file, first_line_first_file);

        // Read all data files to determine the domain bounds
        int layers_to_read = std::min(number_of_layers, _t_inputs.temp_files_in_series); // was given in input file
        for (int layer_read_count = 1; layer_read_count <= layers_to_read; layer_read_count++) {

            std::string tempfile_thislayer = _t_inputs.temp_paths[layer_read_count - 1];
            // binary file type uses extension .catemp, all other file types assumed to be comma-separated ASCII input
            bool binary_input_data = checkTemperatureFileFormat(tempfile_thislayer);
            // { x_min, x_max, y_min, y_max, z_min, z_max }
            std::array<double, 6> xyz_min_max_this_layer =
                parseTemperatureCoordinateMinMax(tempfile_thislayer, binary_input_data);

            // Based on the input file's layer offset, adjust z_min/z_max from the temperature data coordinate
            // system to the multilayer CA coordinate system Check to see in the XYZ bounds for this layer are
            // also limiting for the entire multilayer CA coordinate system
            xyz_min_max_this_layer[4] += deltax * layer_height * (layer_read_count - 1);
            xyz_min_max_this_layer[5] += deltax * layer_height * (layer_read_count - 1);
            if (xyz_min_max_this_layer[0] < x_min)
                x_min = xyz_min_max_this_layer[0];
            if (xyz_min_max_this_layer[1] > x_max)
                x_max = xyz_min_max_this_layer[1];
            if (xyz_min_max_this_layer[2] < y_min)
                y_min = xyz_min_max_this_layer[2];
            if (xyz_min_max_this_layer[3] > y_max)
                y_max = xyz_min_max_this_layer[3];
            if (xyz_min_max_this_layer[4] < z_min)
                z_min = xyz_min_max_this_layer[4];
            if (xyz_min_max_this_layer[5] > z_max)
                z_max = xyz_min_max_this_layer[5];
            z_min_layer[layer_read_count - 1] = xyz_min_max_this_layer[4];
            z_max_layer[layer_read_count - 1] = xyz_min_max_this_layer[5];
            if (id == 0)
                std::cout << "Layer = " << layer_read_count << " Z Bounds are " << xyz_min_max_this_layer[4] << " "
                          << xyz_min_max_this_layer[5] << std::endl;
        }
        // Extend domain in Z (build) direction if the number of layers are simulated is greater than the number
        // of temperature files read
        if (number_of_layers > _t_inputs.temp_files_in_series) {
            for (int layer_read_count = _t_inputs.temp_files_in_series; layer_read_count < number_of_layers;
                 layer_read_count++) {
                if (_t_inputs.temp_files_in_series == 1) {
                    // Only one temperature file was read, so the upper Z bound should account for an additional
                    // "number_of_layers-1" worth of data Since all layers have the same temperature data, each
                    // layer's "z_min_layer" is just translated from that of the first layer
                    z_min_layer[layer_read_count] = z_min_layer[layer_read_count - 1] + deltax * layer_height;
                    z_max_layer[layer_read_count] = z_max_layer[layer_read_count - 1] + deltax * layer_height;
                    z_max += deltax * layer_height;
                }
                else {
                    // "temp_files_in_series" temperature files was read, so the upper Z bound should account for
                    // an additional "number_of_layers-temp_files_in_series" worth of data
                    int repeated_file = (layer_read_count) % _t_inputs.temp_files_in_series;
                    int repeat_unit = layer_read_count / _t_inputs.temp_files_in_series;
                    z_min_layer[layer_read_count] = z_min_layer[repeated_file] + repeat_unit *
                                                                                     _t_inputs.temp_files_in_series *
                                                                                     deltax * layer_height;
                    z_max_layer[layer_read_count] = z_max_layer[repeated_file] + repeat_unit *
                                                                                     _t_inputs.temp_files_in_series *
                                                                                     deltax * layer_height;
                    z_max += deltax * layer_height;
                }
            }
        }

        // If translating temperature data, or if explicit x and y bounds were given, modify the bounds from the
        // temperature data accordingly
        x_max = x_max + (_t_inputs.number_of_copies - 1) * _t_inputs.x_offset;
        y_max = y_max + (_t_inputs.number_of_copies - 1) * _t_inputs.y_offset;
        // If domain bounds were specified, override domain bounds from the data (x_min, x_max, y_min, y_max) in
        // the files with the user specified values
        checkOverrideXYZBounds(id, _s_inputs, _t_inputs);

        // Number of cells in X, Y, and Z of the global simulation domain as calculated from the bounds
        nx = Kokkos::round((x_max - x_min) / deltax) + 1;
        ny = Kokkos::round((y_max - y_min) / deltax) + 1;
        nz = Kokkos::round((z_max - z_min) / deltax) + 1;
    }

    int getDomainSizeAllLayers() {
        int domain_size_all_layers_local = nx * ny_local * nz;
        return domain_size_all_layers_local;
    }

    // Determine the MPI rank of the processor in the +Y direction
    int getNeighborRankNorth(const int id, const int np) {
        int neighbor_rank_north_local;
        if (np > 1) {
            neighbor_rank_north_local = id + 1;
            if (id == np - 1) {
                if (is_periodic)
                    neighbor_rank_north_local = 0;
                else
                    neighbor_rank_north_local = MPI_PROC_NULL;
            }
        }
        else {
            // No MPI communication
            neighbor_rank_north_local = MPI_PROC_NULL;
        }
        return neighbor_rank_north_local;
    }

    // Determine the MPI rank of the processor in the -Y direction
    int getNeighborRankSouth(const int id, const int np) {
        int neighbor_rank_south_local;
        if (np > 1) {
            neighbor_rank_south_local = id - 1;
            if (id == 0) {
                if (is_periodic)
                    neighbor_rank_south_local = np - 1;
                else
                    neighbor_rank_south_local = MPI_PROC_NULL;
            }
        }
        else {
            // No MPI communication
            neighbor_rank_south_local = MPI_PROC_NULL;
        }
        return neighbor_rank_south_local;
    }

    // Determine if this MPI rank is at the domain boundary in the +Y direction
    bool getAtNorthBoundary() {
        bool at_north_boundary_local;
        if (neighbor_rank_north == MPI_PROC_NULL)
            at_north_boundary_local = true;
        else
            at_north_boundary_local = false;
        return at_north_boundary_local;
    }

    // Determine if this MPI rank is at the domain boundary in the -Y direction
    bool getAtSouthBoundary() {
        bool at_south_boundary_local;
        if (neighbor_rank_south == MPI_PROC_NULL)
            at_south_boundary_local = true;
        else
            at_south_boundary_local = false;
        return at_south_boundary_local;
    }

    // Subdivide ny into ny_local across np ranks as evenly as possible, return ny_local on rank id
    int getNyLocal(const int id, const int np) const {
        int ny_local_local = 0;
        int ny_local_est = ny / np;
        int y_remainder = ny % np;
        if (y_remainder == 0) {
            ny_local_local = ny_local_est;
        }
        else {
            if (y_remainder > id) {
                ny_local_local = ny_local_est + 1;
            }
            else {
                ny_local_local = ny_local_est;
            }
        }
        return ny_local_local;
    }

    // Get the offset in the Y direction of the local grid on rank id from the global grid of np ranks, which has ny
    // total cells in Y
    int getYOffset(const int id, const int np) const {
        int y_offset_local = 0;
        int y_offset_est = ny / np;
        int y_remainder = ny % np;
        if (y_remainder == 0) {
            y_offset_local = id * y_offset_est;
        }
        else {
            if (y_remainder > id) {
                y_offset_local = id * (y_offset_est + 1);
            }
            else {
                y_offset_local =
                    (y_offset_est + 1) * (y_remainder - 1) + y_offset_est + 1 + (id - y_remainder) * y_offset_est;
            }
        }
        return y_offset_local;
    }

    // Add ghost nodes to the appropriate subdomains (added where the subdomains overlap, but not at edges of physical
    // domain)
    void addHalo(const int np) {

        // Add halo regions in Y direction if this subdomain borders subdomains on other processors
        // If only 1 rank in the y direction, no halo regions - subdomain is coincident with overall simulation domain
        // If multiple ranks in the y direction, either 1 halo region (borders another rank's subdomain in either the +y
        // or -y direction) or 2 halo regions (if it borders other rank's subdomains in both the +y and -y directions)
        if (neighbor_rank_north != MPI_PROC_NULL)
            ny_local++;
        if (neighbor_rank_south != MPI_PROC_NULL) {
            ny_local++;
            // Also adjust subdomain offset, as these ghost nodes were added on the -y side of the subdomain
            y_offset--;
        }
        // If periodic boundaries and only 1 MPI rank, special case where despite having no MPI neighbors, the domain is
        // extended in Y
        if ((is_periodic) && (np == 1)) {
            ny_local += 2;
            y_offset--;
        }
    }

    // Get the Z coordinate of the lower bound of iteration for layer layernumber
    int calcZLayerBottom(const std::string simulation_type, const int layernumber) {

        int z_layer_bottom_local;
        if ((simulation_type == "FromFile") || (simulation_type == "FromFinch")) {
            // lower bound of domain is based on the data read from the file(s) or from Finch
            z_layer_bottom_local = Kokkos::round((z_min_layer[layernumber] - z_min) / deltax);
        }
        else {
            // Not a multilayer problem, top of "layer" is the top of the overall simulation domain
            z_layer_bottom_local = 0;
        }
        return z_layer_bottom_local;
    }

    // Get the Z coordinate of the upper bound of iteration for layer layernumber
    int calcZLayerTop(const std::string simulation_type, const int layernumber) {

        int z_layer_top_local;
        if ((simulation_type == "FromFile") || (simulation_type == "FromFinch")) {
            // Top of layer comes from the layer's file data or from Finch
            z_layer_top_local = Kokkos::round((z_max_layer[layernumber] - z_min) / deltax);
        }
        else {
            // Not a multilayer problem, top of "layer" is the top of the overall simulation domain
            z_layer_top_local = nz - 1;
        }
        return z_layer_top_local;
    }

    // Calculate the size of the active domain in Z for layer layernumber
    int calcNzLayer(const int id, const int layernumber) {
        int nz_layer_local = z_layer_top - z_layer_bottom + 1;
        if (id == 0)
            std::cout << "Layer " << layernumber << "'s active domain is from Z = " << z_layer_bottom << " through "
                      << z_layer_top << " (" << nz_layer_local << ") cells" << std::endl;
        return nz_layer_local;
    }

    // Calculate the size of the domain, as a number of cells, for this layer of a multilayer problem
    int calcDomainSize() {
        int domain_size_local = nx * ny_local * nz_layer;
        return domain_size_local;
    }

    // Get 1D coordinate corresponding to the first cell in this layer of a multilayer problem
    int getBottomOfCurrentLayer() {
        int bottom_of_current_layer_local = z_layer_bottom * nx * ny_local;
        return bottom_of_current_layer_local;
    }

    // Get 1D coordinate corresponding to the cell after the last cell in this layer of a multilayer problem
    int getTopOfCurrentLayer() {
        int top_of_current_layer_local = z_layer_bottom * nx * ny_local + domain_size;
        return top_of_current_layer_local;
    }

    // Determine new active cell domain size and offset from bottom of global domain
    void initNextLayer(const int id, const std::string simulation_type, const int next_layer_number) {
        if (id == 0)
            std::cout << "Initializing layer " << next_layer_number << std::endl;

        z_layer_bottom = calcZLayerBottom(simulation_type, next_layer_number);
        z_layer_top = calcZLayerTop(simulation_type, next_layer_number);
        nz_layer = calcNzLayer(id, next_layer_number);
        domain_size = calcDomainSize();
        bottom_of_current_layer = getBottomOfCurrentLayer();
        top_of_current_layer = getTopOfCurrentLayer();
        layer_range = std::make_pair(bottom_of_current_layer, top_of_current_layer);
    }

    // Get the 1D cell coordinate from the x, y, and z cell positions of a neighboring cell, returning -1 is the
    // neighbor coordinate is not in bounds
    KOKKOS_INLINE_FUNCTION
    int getNeighbor1DIndex(const int neighbor_coord_x, const int neighbor_coord_y, const int neighbor_coord_z) const {
        int neighbor_index;
        if ((neighbor_coord_x < 0) || (neighbor_coord_x >= nx) || (neighbor_coord_y < 0) ||
            (neighbor_coord_y >= ny_local) || (neighbor_coord_z >= nz_layer) || (neighbor_coord_z < 0))
            neighbor_index = -1;
        else
            neighbor_index = neighbor_coord_z * nx * ny_local + neighbor_coord_x * ny_local + neighbor_coord_y;
        return neighbor_index;
    }

    // Get the 1D cell coordinate from the x, y, and z cell positions
    KOKKOS_INLINE_FUNCTION
    int get1DIndex(const int coord_x, const int coord_y_local, const int coord_z) const {
        int index = coord_z * nx * ny_local + coord_x * ny_local + coord_y_local;
        return index;
    }
    // Get the z cell position of the cell from the 1D cell coordinate
    KOKKOS_INLINE_FUNCTION
    int getCoordZ(const int index) const {
        int coord_z = index / (nx * ny_local);
        return coord_z;
    }
    // Get the y cell position of the cell from the 1D cell coordinate
    KOKKOS_INLINE_FUNCTION
    int getCoordY(const int index) const {
        int rem = index % (nx * ny_local);
        int coord_y = rem % ny_local;
        return coord_y;
    }
    // Get the x cell position of the cell from the 1D cell coordinate
    KOKKOS_INLINE_FUNCTION
    int getCoordX(const int index) const {
        int rem = index % (nx * ny_local);
        int coord_x = rem / ny_local;
        return coord_x;
    }
    // Get the x,y,z position of the cell from the 1D coordinate
    KOKKOS_FUNCTION
    void getCoordXYZ(int (&cell_location)[3], const int index) const {
        cell_location[2] = getCoordZ(index);
        int rem = index % (nx * ny_local);
        cell_location[1] = rem % ny_local;
        cell_location[0] = rem / ny_local;
    }
    // Get the z cell position of the cell from the 1D cell coordinate with respect to the overall simulation domain
    // (all MPI ranks)
    KOKKOS_INLINE_FUNCTION
    int getCoordZGlobal(const int index) const {
        int coord_z = index / (nx * ny);
        return coord_z;
    }
    // Get the y cell position of the cell from the 1D cell coordinate with respect to the overall simulation domain
    // (all MPI ranks)
    KOKKOS_INLINE_FUNCTION
    int getCoordYGlobal(const int index, const bool is_periodic = false) const {
        int nx_adj, ny_adj;
        if (is_periodic) {
            nx_adj = nx - 2;
            ny_adj = ny - 2;
        }
        else {
            nx_adj = nx;
            ny_adj = ny;
        }
        int rem = index % (nx_adj * ny_adj);
        int coord_y = rem % ny_adj + is_periodic;
        return coord_y;
    }
    // Get the x cell position of the cell from the 1D cell coordinate with respect to the overall simulation domain
    // (all MPI ranks)
    KOKKOS_INLINE_FUNCTION
    int getCoordXGlobal(const int index, const bool is_periodic = false) const {
        int nx_adj, ny_adj;
        if (is_periodic) {
            nx_adj = nx - 2;
            ny_adj = ny - 2;
        }
        else {
            nx_adj = nx;
            ny_adj = ny;
        }
        int rem = index % (nx_adj * ny_adj);
        int coord_x = rem / ny_adj + is_periodic;
        return coord_x;
    }
};

#endif
